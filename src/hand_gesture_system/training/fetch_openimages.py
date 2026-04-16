from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

OPEN_IMAGES_SPLITS = {
    "train": {
        "labels_url": "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv",
        "metadata_url": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    },
    "validation": {
        "labels_url": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
        "metadata_url": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    },
    "test": {
        "labels_url": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv",
        "metadata_url": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    },
}
CLASS_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
DEFAULT_CLASS_NAME = "Human hand"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return value.strip("_") or "dataset"


def _stream_csv_rows(
    session: requests.Session,
    *,
    url: str,
    timeout: float,
):
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        lines = (
            line.decode("utf-8", errors="ignore")
            for line in response.iter_lines()
            if line
        )
        reader = csv.DictReader(lines)
        for row in reader:
            yield row


def _resolve_class_id(
    session: requests.Session,
    *,
    class_name: str,
    timeout: float,
) -> str:
    target = class_name.strip().lower()
    if not target:
        raise ValueError("Class name cannot be empty")

    with session.get(CLASS_DESCRIPTIONS_URL, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        lines = (
            line.decode("utf-8", errors="ignore")
            for line in response.iter_lines()
            if line
        )
        reader = csv.DictReader(lines)
        for row in reader:
            display_name = str(row.get("DisplayName", "")).strip()
            if display_name.lower() == target:
                return str(row["LabelName"]).strip()

    raise ValueError(
        f"Class '{class_name}' not found in Open Images class descriptions."
    )


def _collect_positive_ids(
    session: requests.Session,
    *,
    split: str,
    class_id: str,
    timeout: float,
    max_ids: int,
) -> list[str]:
    if max_ids <= 0:
        return []

    labels_url = OPEN_IMAGES_SPLITS[split]["labels_url"]
    image_ids: list[str] = []
    seen: set[str] = set()

    for row in _stream_csv_rows(session, url=labels_url, timeout=timeout):
        if row.get("LabelName") != class_id:
            continue
        if str(row.get("Confidence", "")).strip() != "1":
            continue

        image_id = str(row.get("ImageID", "")).strip()
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)

        if len(image_ids) >= max_ids:
            break

    return image_ids


def _pick_url(row: dict[str, str], prefer: str) -> str:
    if prefer == "thumbnail":
        thumb = str(row.get("Thumbnail300KURL", "")).strip()
        if thumb:
            return thumb
    return str(row.get("OriginalURL", "")).strip()


def _infer_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    return ".jpg"


def _download_image(
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

    image = cv2.imread(str(image_path))
    if image is None:
        return False
    return bool(tracker.detect(image))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download hand images directly from Open Images (an AI training dataset)"
    )
    parser.add_argument("--output-dir", default="data/openimages")
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--class-id", default=None, help="Override class MID (e.g. /m/0k65p)")
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="Comma-separated: train,validation,test",
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument(
        "--url-field",
        choices=["thumbnail", "original"],
        default="thumbnail",
        help="Image URL source from metadata CSV",
    )
    parser.add_argument("--oversample-factor", type=float, default=1.7)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-hand", action="store_true")
    parser.add_argument("--verify-max-hands", type=int, default=2)
    parser.add_argument("--verify-min-confidence", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.oversample_factor < 1.0:
        raise ValueError("--oversample-factor must be >= 1.0")

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("No splits selected")
    for split in splits:
        if split not in OPEN_IMAGES_SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Use train,validation,test.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "hand-gesture-system/0.1 (Open Images downloader)"
        }
    )

    class_id = args.class_id.strip() if args.class_id else _resolve_class_id(
        session, class_name=args.class_name, timeout=args.timeout
    )
    class_name = args.class_name.strip()
    class_slug = _slugify(class_name)

    out_dir = Path(args.output_dir) / class_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if args.overwrite:
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "manifest.jsonl"):
            for item in out_dir.glob(pattern):
                item.unlink()

    existing_images = sorted(
        [
            *out_dir.glob("*.jpg"),
            *out_dir.glob("*.jpeg"),
            *out_dir.glob("*.png"),
            *out_dir.glob("*.webp"),
        ]
    )
    existing_count = len(existing_images)
    if existing_count >= args.limit:
        print(f"Already have {existing_count} images in {out_dir}; nothing to do.")
        return 0

    to_download = args.limit - existing_count
    candidate_target = int(to_download * args.oversample_factor)

    print(
        f"Open Images class: {class_name} ({class_id}) | "
        f"splits={splits} | target_new={to_download}"
    )

    ids_by_split: dict[str, list[str]] = {}
    remaining = candidate_target
    for split in splits:
        if remaining <= 0:
            ids_by_split[split] = []
            continue

        positive_ids = _collect_positive_ids(
            session,
            split=split,
            class_id=class_id,
            timeout=args.timeout,
            max_ids=remaining,
        )
        ids_by_split[split] = positive_ids
        remaining -= len(positive_ids)
        print(f"[{split}] collected candidate IDs: {len(positive_ids)}")

    all_candidates = [
        (split, image_id)
        for split, split_ids in ids_by_split.items()
        for image_id in split_ids
    ]
    if not all_candidates:
        raise RuntimeError(
            "No candidate image IDs found. Try different splits/class."
        )

    random.Random(args.shuffle_seed).shuffle(all_candidates)
    wanted_by_split: dict[str, set[str]] = {split: set() for split in splits}
    for split, image_id in all_candidates:
        if sum(len(v) for v in wanted_by_split.values()) >= candidate_target:
            break
        wanted_by_split[split].add(image_id)

    tracker = None
    if args.verify_hand:
        from hand_gesture_system.tracking.mediapipe_tracker import MediaPipeHandTracker

        tracker = MediaPipeHandTracker(
            max_hands=args.verify_max_hands,
            static_image_mode=True,
            min_detection_confidence=args.verify_min_confidence,
            min_tracking_confidence=args.verify_min_confidence,
        )

    downloaded = 0
    rejected = 0
    file_index = existing_count

    try:
        for split in splits:
            wanted_ids = wanted_by_split[split]
            if not wanted_ids:
                continue

            metadata_url = OPEN_IMAGES_SPLITS[split]["metadata_url"]
            for row in _stream_csv_rows(session, url=metadata_url, timeout=args.timeout):
                image_id = str(row.get("ImageID", "")).strip()
                if image_id not in wanted_ids:
                    continue

                image_url = _pick_url(row, prefer=args.url_field)
                if not image_url:
                    wanted_ids.remove(image_id)
                    continue

                file_index += 1
                extension = _infer_extension(image_url)
                file_name = f"{class_slug}_{file_index:06d}_{image_id}{extension}"
                out_path = out_dir / file_name

                ok = _download_image(
                    session,
                    url=image_url,
                    out_path=out_path,
                    timeout=args.timeout,
                )
                if not ok:
                    wanted_ids.remove(image_id)
                    continue

                if tracker is not None and not _contains_hand(out_path, tracker):
                    out_path.unlink(missing_ok=True)
                    rejected += 1
                    wanted_ids.remove(image_id)
                    continue

                payload = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "split": split,
                    "image_id": image_id,
                    "url": image_url,
                    "license": row.get("License"),
                    "title": row.get("Title"),
                    "file": file_name,
                }
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")

                downloaded += 1
                wanted_ids.remove(image_id)
                print(f"saved {downloaded + existing_count}/{args.limit}: {file_name}")

                if downloaded >= to_download:
                    break

            if downloaded >= to_download:
                break

    finally:
        if tracker is not None:
            tracker.close()

    final_count = len(
        [
            *out_dir.glob("*.jpg"),
            *out_dir.glob("*.jpeg"),
            *out_dir.glob("*.png"),
            *out_dir.glob("*.webp"),
        ]
    )
    print(
        f"\nDone. Saved {downloaded} new images (total={final_count}) to {out_dir}"
        + (f"; rejected_non_hand={rejected}" if args.verify_hand else "")
    )
    if final_count < args.limit:
        print(
            "Warning: target not fully reached. Increase --oversample-factor "
            "or include more splits."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
