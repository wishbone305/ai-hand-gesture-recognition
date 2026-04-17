# PyInstaller spec — Hand Gesture Recognition Demo
#
# Build:
#   pip install pyinstaller
#   cd /path/to/ai\ hand\ gesture\ recognition
#   pyinstaller demo.spec
#
# Output: dist/GestureDemo/GestureDemo  (run this)
# For a single .exe flip ONEFILE = True (slower to launch, easier to share)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
from pathlib import Path

ONEFILE = False
HERE    = Path(SPECPATH)
SRC     = HERE / "src"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
datas = []

# MediaPipe bundled data (task models, tflite files, proto descriptors)
datas += collect_data_files("mediapipe")

# Trained small model (1 M params) + labels
_small = HERE / "artifacts" / "stgcn_v2" / "stgcn_small"
datas += [
    (str(_small / "best_model.pt"), "artifacts/stgcn_v2/stgcn_small"),
    (str(_small / "labels.txt"),    "artifacts/stgcn_v2/stgcn_small"),
]

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
hiddenimports = [
    "hand_gesture_system",
    "hand_gesture_system.training.stgcn_model",
    "torch", "torch.nn", "torch.nn.functional",
    "mediapipe",
    "mediapipe.tasks", "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "mediapipe.tasks.python.core",
    "mediapipe.tasks.python.core.base_options",
    "numpy", "cv2",
    "urllib.request", "collections", "pathlib",
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
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
              name="GestureDemo", debug=False, strip=False,
              upx=True, console=False)
else:
    exe = EXE(pyz, a.scripts, name="GestureDemo",
              debug=False, strip=False, upx=False, console=False)
    coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas,
                   name="GestureDemo", strip=False, upx=False)
