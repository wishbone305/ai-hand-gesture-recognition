from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_CLASS_QUERIES = {
    "open_palm": "open hand palm gesture person",
    "fist": "clenched fist hand gesture",
    "peace": "peace sign hand gesture",
    "thumbs_up": "thumbs up hand gesture",
    "pointing": "index finger pointing gesture",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_class_query(text: str) -> tuple[str, str]:
    for separator in (":", "="):
        if separator in text:
            label, query = text.split(separator, 1)
            label = label.strip()
            query = query.strip()
            if not label or not query:
                break
            return label, query
    raise ValueError(
        f"Invalid --class-query '{text}'. Expected format 'label:search terms'"
    )


def _extension_from_url(url: str, mime: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix

    if "png" in mime:
        return ".png"
    if "webp" in mime:
        return ".webp"
    return ".jpg"


def _iter_wikimedia_candidates(
    session: requests.Session,
    *,
    query: str,
    thumb_width: int,
    max_pages: int,
    timeout: float,
):
    continue_params: dict[str, object] = {}

    for _ in range(max_pages):
        params: dict[str, object] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": thumb_width,
        }
        params.update(continue_params)

        response = session.get(WIKIMEDIA_API, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue

            info = info_list[0]
            url = str(info.get("thumburl") or info.get("url") or "")
            if not url:
                continue

            yield {
                "title": str(page.get("title") or ""),
                "url": url,
                "width": int(info.get("thumbwidth") or info.get("width") or 0),
                "height": int(info.get("thumbheight") or info.get("height") or 0),
                "mime": str(info.get("mime") or ""),
            }

        if "continue" not in payload:
            break
        continue_params = payload["continue"]


def _download_file(
    session: requests.Session,
    *,
    url: str,
    out_path: Path,
    timeout: float,
) -> bool:
    response = session.get(url, timeout=timeout)
    if response.status_code != 200:
        return False

    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "image" not in content_type:
        return False

    out_path.write_bytes(response.content)
    return True


def _contains_hand(image_path: Path, tracker) -> bool:
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        return False
    return bool(tracker.detect(frame))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download hand gesture images from Wikimedia Commons for dataset bootstrapping"
    )
    parser.add_argument("--output-dir", default="data/web_images")
    parser.add_argument(
        "--class-query",
        action="append",
        default=[],
        help="Repeatable format: label:search query",
    )
    parser.add_argument(
        "--use-default-gestures",
        action="store_true",
        help="Use built-in gesture queries (open_palm, fist, peace, thumbs_up, pointing)",
    )
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--min-width", type=int, default=224)
    parser.add_argument("--min-height", type=int, default=224)
    parser.add_argument("--thumb-width", type=int, default=640)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-hand",
        action="store_true",
        help="Use MediaPipe to keep only images where a hand is detected",
    )
    parser.add_argument("--verify-max-hands", type=int, default=2)
    parser.add_argument("--verify-min-confidence", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.per_class < 1:
        raise ValueError("--per-class must be >= 1")
    if args.max_pages < 1:
        raise ValueError("--max-pages must be >= 1")
    if args.min_width < 1 or args.min_height < 1:
        raise ValueError("--min-width/--min-height must be >= 1")
    if args.thumb_width < 64:
        raise ValueError("--thumb-width must be >= 64")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be >= 0")

    class_queries: list[tuple[str, str]] = []
    for item in args.class_query:
        class_queries.append(_parse_class_query(item))
    if args.use_default_gestures:
        class_queries.extend(DEFAULT_CLASS_QUERIES.items())
    if not class_queries:
        raise ValueError(
            "Provide at least one --class-query or use --use-default-gestures"
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "hand-gesture-system/0.1 "
                "(dataset bootstrap; https://commons.wikimedia.org/)"
            )
        }
    )

    hand_tracker = None
    if args.verify_hand:
        from hand_gesture_system.tracking.mediapipe_tracker import MediaPipeHandTracker

        hand_tracker = MediaPipeHandTracker(
            max_hands=args.verify_max_hands,
            static_image_mode=True,
            min_detection_confidence=args.verify_min_confidence,
            min_tracking_confidence=args.verify_min_confidence,
        )

    total_saved = 0
    total_rejected = 0

    try:
        for label, query in class_queries:
            label_dir = output_root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = label_dir / "metadata.jsonl"

            if args.overwrite:
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "metadata.jsonl"):
                    for item in label_dir.glob(pattern):
                        item.unlink()

            existing_images = sorted(
                [
                    *label_dir.glob("*.jpg"),
                    *label_dir.glob("*.jpeg"),
                    *label_dir.glob("*.png"),
                    *label_dir.glob("*.webp"),
                ]
            )
            saved_count = len(existing_images)

            seen_urls: set[str] = set()
            if metadata_path.exists():
                for line in metadata_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        url = str(row.get("url", ""))
                        if url:
                            seen_urls.add(url)
                    except json.JSONDecodeError:
                        continue

            print(f"\n[{label}] query='{query}' target={args.per_class} existing={saved_count}")

            for candidate in _iter_wikimedia_candidates(
                session,
                query=query,
                thumb_width=args.thumb_width,
                max_pages=args.max_pages,
                timeout=args.timeout,
            ):
                if saved_count >= args.per_class:
                    break

                url = str(candidate["url"])
                width = int(candidate["width"])
                height = int(candidate["height"])
                mime = str(candidate["mime"])

                if url in seen_urls:
                    continue
                if width < args.min_width or height < args.min_height:
                    continue
                if not mime.startswith("image/"):
                    continue

                extension = _extension_from_url(url, mime)
                file_name = f"{label}_{saved_count + 1:05d}{extension}"
                out_path = label_dir / file_name

                ok = _download_file(
                    session,
                    url=url,
                    out_path=out_path,
                    timeout=args.timeout,
                )
                if not ok:
                    continue

                if hand_tracker is not None and not _contains_hand(out_path, hand_tracker):
                    out_path.unlink(missing_ok=True)
                    total_rejected += 1
                    continue

                row = {
                    "label": label,
                    "query": query,
                    "title": candidate["title"],
                    "url": url,
                    "width": width,
                    "height": height,
                    "mime": mime,
                    "file": out_path.name,
                }
                with metadata_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")

                seen_urls.add(url)
                saved_count += 1
                total_saved += 1
                print(f"[{label}] saved {saved_count}/{args.per_class}: {out_path.name}")

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            if saved_count < args.per_class:
                print(
                    f"[{label}] warning: only collected {saved_count}/{args.per_class}. "
                    "Increase --max-pages or adjust query."
                )

    finally:
        if hand_tracker is not None:
            hand_tracker.close()

    print(
        f"\nDone. Downloaded {total_saved} images"
        + (
            f", rejected {total_rejected} as non-hand."
            if args.verify_hand
            else "."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
