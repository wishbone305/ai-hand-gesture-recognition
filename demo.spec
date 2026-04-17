# PyInstaller spec for the Hand Gesture Recognition demo.
#
# Build with:
#   cd /path/to/ai\ hand\ gesture\ recognition
#   pyinstaller demo.spec
#
# Output: dist/GestureDemo  (macOS .app-style folder) or dist/GestureDemo.exe on Windows
# For a true single file: change BUNDLE_ONEFILE to True below.

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

BUNDLE_ONEFILE = False   # True = single .exe / .app (slow to start); False = folder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(SPECPATH)
SRC  = HERE / "src"

# ---------------------------------------------------------------------------
# Data files to bundle
# ---------------------------------------------------------------------------
datas = []

# MediaPipe model files (mediapipe ships them inside its package)
datas += collect_data_files("mediapipe")

# Trained model checkpoints — include every best_model.pt found
import glob
for pt in glob.glob(str(HERE / "artifacts" / "stgcn_v2" / "**" / "best_model.pt"), recursive=True):
    rel = str(Path(pt).parent.relative_to(HERE))
    datas.append((pt, f"artifacts/stgcn_v2/{Path(pt).parent.name}"))

# labels.txt files alongside each checkpoint
for txt in glob.glob(str(HERE / "artifacts" / "stgcn_v2" / "**" / "labels.txt"), recursive=True):
    datas.append((txt, f"artifacts/stgcn_v2/{Path(txt).parent.name}"))

# ---------------------------------------------------------------------------
# Hidden imports (things PyInstaller misses via static analysis)
# ---------------------------------------------------------------------------
hiddenimports = [
    # hand_gesture_system package
    "hand_gesture_system",
    "hand_gesture_system.training.stgcn_model",
    # torch / mediapipe internals
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision",
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "mediapipe.tasks.python.core",
    "mediapipe.tasks.python.core.base_options",
    # numpy / cv2
    "numpy",
    "cv2",
    # stdlib
    "urllib.request",
    "collections",
    "pathlib",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["demo.py"],
    pathex=[str(SRC)],
    binaries=collect_dynamic_libs("mediapipe") + collect_dynamic_libs("cv2"),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython"],
    noarchive=False,
)

pyz = PYZ(a.pure)

if BUNDLE_ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
        name="GestureDemo",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,    # no terminal window on macOS/Windows
        icon=None,
    )
else:
    exe = EXE(
        pyz, a.scripts,
        name="GestureDemo",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        name="GestureDemo",
        strip=False,
        upx=False,
    )
