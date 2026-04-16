# AI Hand Gesture Recognition (Model-Agnostic + Temporal Upgrade)

This project now includes a full model-agnostic hand gesture pipeline with:

- Hand mesh mapping (21 landmarks) via MediaPipe.
- Rule-based gestures and finger counting.
- Static model adapters (`scikit-learn`, `ONNX`, `TorchScript`, HTTP).
- Temporal sequence features (`x`, `dx`, `dx2`) for dynamic gestures/signs.
- Dynamic sequence model adapters (`ONNX`, `TorchScript`).
- DTW template fallback for low-data dynamic gesture classes.
- Prediction smoothing/debounce + confusion-fix postprocessing.
- Benchmark tooling for latency + per-class precision/recall/F1.

## Project layout

- `tracking/*`: hand tracking.
- `features/*`: static and temporal feature extraction.
- `rules/*`: rule-based gesture recognition.
- `models/*`: static + temporal adapters, including DTW fallback.
- `postprocessing/*`: smoothing and confusion fixes.
- `training/*`: dataset collection/export, training, evaluation, model export.
- `pipeline.py`: orchestration, hand identity tracking, prediction fusion.
- `cli.py`: real-time webcam app.
- `benchmark.py`: adapter benchmark utility.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run in rule-only mode:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli --model-type none
```

Press `q` or `Esc` to exit.

Live UI behavior:

- Top-left summary box shows detected finger count + gesture per hand.
- Top-right `REC` button toggles recording (click button or press `r`).
- Recordings are saved to `recordings/` by default.

## Test with a single image

Rule-only test on one image:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.image_test \
  --image /path/to/hand.jpg \
  --save-annotated outputs/hand_annotated.jpg
```

With a trained static model backend:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.image_test \
  --image /path/to/hand.jpg \
  --model-type onnx \
  --model-path /path/to/static_model.onnx \
  --labels-path /path/to/static_labels.txt \
  --save-annotated outputs/hand_annotated.jpg
```

The command prints per-hand predictions and can optionally display the image via `--show`.

## Pull hundreds of images from the internet

Use Wikimedia Commons search to bootstrap datasets:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.fetch_images \
  --output-dir data/web_images \
  --class-query open_palm:\"open hand palm gesture\" \
  --class-query fist:\"clenched fist hand gesture\" \
  --per-class 200 \
  --max-pages 30
```

Or use built-in default gesture queries (5 classes):

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.fetch_images \
  --output-dir data/web_images \
  --use-default-gestures \
  --per-class 200 \
  --max-pages 30
```

Optional quality filter:

```bash
... --verify-hand
```

`--verify-hand` runs MediaPipe detection and keeps only images where at least one hand is found.

### Pull from Open Images (AI training dataset)

Open Images is a large-scale dataset used widely in AI training pipelines. Download hundreds of `Human hand` images:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.fetch_openimages \
  --output-dir data/openimages \
  --class-name "Human hand" \
  --splits validation,test \
  --limit 400 \
  --url-field thumbnail
```

Optional strict filtering:

```bash
... --verify-hand
```

## Static model backends

### scikit-learn

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --model-type sklearn \
  --model-path /path/to/model.joblib
```

### ONNX

```bash
pip install onnxruntime
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --model-type onnx \
  --model-path /path/to/static_model.onnx \
  --labels-path /path/to/static_labels.txt
```

### TorchScript

```bash
pip install torch
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --model-type torchscript \
  --model-path /path/to/static_model.pt \
  --labels-path /path/to/static_labels.txt
```

### HTTP model server

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --model-type http \
  --http-endpoint http://localhost:8000/predict
```

Set API key if needed:

```bash
export GESTURE_API_KEY=your_token
```

## Dynamic sequence backends

### ONNX sequence model

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --dynamic-model-type onnx \
  --dynamic-model-path /path/to/sequence_model.onnx \
  --dynamic-labels-path /path/to/sequence_labels.txt \
  --dynamic-input-layout btd \
  --sequence-length 32 \
  --min-sequence-frames 8
```

### TorchScript sequence model

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --dynamic-model-type torchscript \
  --dynamic-model-path /path/to/sequence_model.pt \
  --dynamic-labels-path /path/to/sequence_labels.txt \
  --dynamic-input-layout btd
```

`--dynamic-input-layout` options:

- `btd`: `[1, frames, features]`
- `td`: `[frames, features]`
- `flat`: `[1, frames*features]`

## DTW fallback (tiny datasets)

DTW templates file example (`templates.json`):

```json
{
  "wave": [
    [[0.1, 0.2], [0.2, 0.3]],
    [[0.1, 0.25], [0.22, 0.31]]
  ],
  "circle": [
    [[0.3, 0.2], [0.4, 0.25]]
  ]
}
```

Use with CLI:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --dtw-templates-path /path/to/templates.json \
  --dtw-distance-threshold 0.35 \
  --dtw-score-scale 6.0 \
  --dtw-top-k 3
```

You can combine static model + dynamic model + DTW fallback in one run.

## Smoothing and confusion-fix controls

Enabled by default.

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --disable-smoothing \
  --disable-confusion-fix
```

Or tune smoothing:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --smoothing-window 8 \
  --smoothing-min-consecutive 3 \
  --smoothing-stable-score 0.9
```

## Benchmark adapters

Dataset format (`.npz`):

- Static: keys `X`, `y`
- Dynamic: keys `X_seq`, `y`

### Static benchmark

```bash
PYTHONPATH=src python3 -m hand_gesture_system.benchmark \
  --dataset-path /path/to/benchmark.npz \
  static \
  --backend onnx \
  --model-path /path/to/static_model.onnx \
  --labels-path /path/to/static_labels.txt
```

### Dynamic benchmark

```bash
PYTHONPATH=src python3 -m hand_gesture_system.benchmark \
  --dataset-path /path/to/benchmark.npz \
  dynamic \
  --backend torchscript \
  --model-path /path/to/sequence_model.pt \
  --labels-path /path/to/sequence_labels.txt \
  --input-layout btd
```

Optional JSON report:

```bash
... --output-json /path/to/report.json
```

## Sequence training pipeline

Install training dependency:

```bash
pip install torch
```

### 1) Collect temporal samples

This stores per-sample temporal feature tensors as `.npy` under `data/sequences/<label>/`.

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.collect \
  --label wave \
  --samples 60 \
  --sequence-length 32 \
  --output-dir data/sequences
```

Controls:

- `Space`: start recording one sample
- `q` / `Esc`: quit

### 2) Optional dataset export to `.npz`

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.export_dataset \
  --dataset-dir data/sequences \
  --output-npz data/sequences_dataset.npz \
  --labels-out data/labels.txt
```

### 3) Train sequence model

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.train \
  --dataset data/sequences \
  --sequence-length 32 \
  --epochs 40 \
  --batch-size 64 \
  --out-dir artifacts/sequence \
  --run-name run_001 \
  --export-torchscript \
  --export-onnx
```

Outputs in `artifacts/sequence/run_001/`:

- `best_model.pt` (training checkpoint + config + labels)
- `labels.txt`
- `training_summary.json`
- `model_sequence.pt` (if `--export-torchscript`)
- `model_sequence.onnx` (if `--export-onnx`)

### 4) Evaluate trained model

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.evaluate \
  --checkpoint artifacts/sequence/run_001/best_model.pt \
  --dataset data/sequences \
  --output-json artifacts/sequence/run_001/eval.json
```

### 5) Export from checkpoint later

```bash
PYTHONPATH=src python3 -m hand_gesture_system.training.export \
  --checkpoint artifacts/sequence/run_001/best_model.pt \
  --out-dir artifacts/sequence/run_001/export \
  --torchscript \
  --onnx
```

### 6) Use exported model in real-time CLI

TorchScript sequence backend:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --dynamic-model-type torchscript \
  --dynamic-model-path artifacts/sequence/run_001/model_sequence.pt \
  --dynamic-labels-path artifacts/sequence/run_001/labels.txt \
  --dynamic-input-layout btd
```

ONNX sequence backend:

```bash
PYTHONPATH=src python3 -m hand_gesture_system.cli \
  --dynamic-model-type onnx \
  --dynamic-model-path artifacts/sequence/run_001/model_sequence.onnx \
  --dynamic-labels-path artifacts/sequence/run_001/labels.txt \
  --dynamic-input-layout btd
```

## Built-in rule labels

- `fingers_0` to `fingers_5`
- `fist`
- `open_palm`
- `peace`
- `thumbs_up`
- `pointing`
- `ok`
- `rock_sign`

## Notes

- For robust sign language recognition, use temporal sequence models.
- DTW fallback is useful for small custom dynamic gesture sets.
- Prefer licensed repositories (Apache/MIT/BSD) when reusing external code patterns.
# ai-hand-gesture-recognition
