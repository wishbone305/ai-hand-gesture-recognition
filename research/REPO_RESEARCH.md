# Hand Gesture + Sign Language Repo Research (20 Cloned Repos)

Research date: 2026-04-15
Local clone root: `research/repos`
Inventory file: `research/repo_inventory.tsv`

## 1) Corpus (20 repos cloned)

1. google-ai-edge/mediapipe
2. Kazuhito00/hand-gesture-recognition-using-mediapipe
3. PINTO0309/hand-gesture-recognition-using-onnx
4. kinivi/tello-gesture-control
5. cvzone/cvzone
6. gabguerin/Sign-Language-Recognition--MediaPipe-DTW
7. kevinjosethomas/sign-language-processing
8. 209sontung/sign-language
9. HarshitDolu/Finger-Counter-using-mediapipe
10. OwenTalmo/finger-counter
11. Viral-Doshi/Gesture-Controlled-Virtual-Mouse
12. wolterlw/hand_tracking
13. xinyang-sun/HandTrackingWithUnity-Mediapipe
14. neural-control-and-computation-lab/athena
15. luc1k1/hand-tracking
16. sorohere/Hand-Pose-Detection
17. ofelipelucca/LibrasController
18. MonzerDev/Real-Time-Sign-Language-Recognition
19. RhythmusByte/Sign-Language-to-Speech
20. Nam-H-Pham/American-Sign-Language-Recognition

## 2) High-signal repos (recommended for integration)

### A. Core hand tracking + gesture baseline
- `google-ai-edge/mediapipe`
  - Why: canonical, maintained, cross-platform task APIs.
  - Use: tracker foundation + model/task portability strategy.
- `Kazuhito00/hand-gesture-recognition-using-mediapipe`
  - Why: strongest practical baseline for static hand signs + temporal point-history gestures.
  - Use: normalization/data-logging/training pipeline for landmarks.
- `PINTO0309/hand-gesture-recognition-using-onnx`
  - Why: full ONNX runtime path, reduced framework lock-in.
  - Use: adapter patterns for ONNX inference providers and model conversion flow.

### B. Sign-language modeling (best ideas)
- `209sontung/sign-language`
  - Why: explicit temporal architecture (1D CNN + Transformer) on landmark sequences.
  - Use: sequence model design and preprocessing features (`x`, `dx`, `dx2`).
- `kevinjosethomas/sign-language-processing`
  - Why: end-to-end system thinking (recognition + language post-processing + expressive output path).
  - Use: stability heuristics, lexical post-processing, service decomposition.
- `gabguerin/Sign-Language-Recognition--MediaPipe-DTW`
  - Why: no-training temporal recognizer via DTW over frame-wise features.
  - Use: low-data fallback recognizer and rapid prototyping for dynamic signs.

### C. Production utility patterns
- `neural-control-and-computation-lab/athena`
  - Why: multi-camera 2D->3D triangulation + smoothing/refinement.
  - Use: future upgrade path to robust 3D hand trajectory modeling.
- `cvzone/cvzone`
  - Why: pragmatic utilities (`fingersUp`, distances, bbox helpers).
  - Use: utility-level ergonomics, not as model backbone.

## 3) Concrete technical findings

### 3.1 Landmark normalization patterns
- Kazuhito/tello pipelines standardize by:
  - wrist-relative coordinates
  - max-abs normalization across flattened coordinates
  - optional point-history normalization by frame size
- This is simple, fast, and robust for static sign classes.

References:
- `research/repos/hand-gesture-recognition-using-mediapipe/app.py`
- `research/repos/tello-gesture-control/gestures/gesture_recognition.py`

### 3.2 Temporal modeling approaches found
- **Point history + small classifier** (Kazuhito): easy real-time dynamic gestures.
- **DTW over angle embeddings** (Sicara-based repo): no heavy training, good for small vocab dynamic signs.
- **1D CNN + Transformer** (`209sontung/sign-language`): stronger scaling path for sign-language sequences.

References:
- `research/repos/hand-gesture-recognition-using-mediapipe/app.py`
- `research/repos/Sign-Language-Recognition--MediaPipe-DTW/models/hand_model.py`
- `research/repos/Sign-Language-Recognition--MediaPipe-DTW/utils/dtw.py`
- `research/repos/sign-language/src/backbone.py`
- `research/repos/sign-language/src/utils.py`

### 3.3 Inference backend portability
- Best portability pattern is ONNX adapter + provider list fallback (CPU/GPU) seen in PINTO repo.
- Keep model contracts shape-stable and isolate backend-specific code in adapters.

References:
- `research/repos/hand-gesture-recognition-using-onnx/model/palm_detection/palm_detection.py`
- `research/repos/hand-gesture-recognition-using-onnx/model/keypoint_classifier/keypoint_classifier.py`

### 3.4 Stability heuristics that matter in UX
- Consecutive-frame voting / debounce before token acceptance.
- Heuristic disambiguation for commonly confused letters (e.g., A/T, D/I, F/W).
- This dramatically reduces jitter for live sign decoding.

Reference:
- `research/repos/sign-language-processing/src/server/utils/recognition.py`

### 3.5 Finger-counting heuristics
- Two robust practical variants found:
  - tip-vs-PIP/MCP y-comparison for fingers + handed thumb x rule
  - use MediaPipe Tasks API for async detector callback flow

Reference:
- `research/repos/finger-counter/main.py`
- `research/repos/cvzone/cvzone/HandTrackingModule.py`

### 3.6 3D/multi-camera path
- Athena shows practical triangulation + Savitzky-Golay smoothing + NaN run handling.
- This is relevant if you later need high-fidelity hand mesh trajectory analysis or multi-view data collection.

Reference:
- `research/repos/athena/athena/triangulaterefine.py`

## 4) Repos to use carefully (risk flags)

### Licensing risk
- `NO-LICENSE` repos are not safe for direct code reuse in distributed software.
- Prefer Apache-2.0 / MIT / BSD-3-Clause repos for direct code transfer.

### Copyleft constraints
- GPL/AGPL repos can impose reciprocal obligations if code is incorporated.
- Keep them as idea references unless your project licensing plan explicitly supports copyleft.

### Staleness/dependency drift
- Several repos pin older MediaPipe/TensorFlow versions.
- Re-implementation of patterns is safer than direct copy-paste.

## 5) Best integration blueprint for your project

For your model-agnostic system, adopt this stack:

1. Tracker: MediaPipe Hands (current baseline) with optional ONNX palm/landmark path.
2. Features:
   - static: wrist-relative normalized landmarks
   - temporal: add velocity and acceleration channels (`dx`, `dx2`) per frame
3. Recognition layers:
   - rule engine for finger count + friendly gestures
   - static sign classifier (MLP/CNN)
   - temporal sign classifier (Transformer/LSTM)
   - DTW fallback for low-data dynamic classes
4. Stabilization:
   - consecutive-frame confirmation
   - class-specific confusion fix rules
5. Model portability:
   - keep adapters for sklearn/ONNX/Torch/HTTP
   - export canonical shape contracts for all backends

## 6) Immediate implementation priorities (highest ROI)

1. Add temporal sequence buffer + `dx/dx2` feature extraction to your current project.
2. Add a sequence classifier adapter (ONNX/Torch) for dynamic signs.
3. Add prediction smoothing/debounce + confusion-fix postprocessor.
4. Add DTW fallback recognizer for custom dynamic gestures with tiny datasets.
5. Build a benchmark script (latency + per-class precision/recall) across adapters.

## 7) Files produced during research

- `research/repo_inventory.tsv`
- `research/REPO_RESEARCH.md`

