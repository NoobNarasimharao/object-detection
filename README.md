# Real-Time Multi-Object Detection and Tracking Using YOLO and ByteTrack

A third-year B.Tech Computer Vision laboratory application with a Streamlit dashboard and reusable CLI. YOLO26 detects COCO objects; explicitly selected ByteTrack associates detections across frames. PyTorch selects CUDA when available, otherwise CPU. Actual real-time throughput depends on hardware, resolution and scene complexity.

## Features

- All 80 pretrained classes, configurable confidence, NMS IoU, model scale and inference resolution.
- Tracker-owned IDs, temporal confirmation, confidence-weighted class history, class colors, bounded trails and per-frame records.
- Unique IDs per class, simultaneous detections, and finite-line crossings counted separately.
- Upload MP4/AVI/MOV/MKV videos, JPG/PNG/JPEG/BMP/WebP images, or use a camera attached to the local server.
- Background dashboard processing, live annotated preview, progress, cancellation and partial output.
- Plotly charts, SQLite history, downloadable video, CSV and JSON.
- Optional independent ByteTrack/BoT-SORT comparison runs through the CLI.

## Environment and installation

Reuse the existing `.venv`. This repository deliberately does not create, replace, or reset environments. Python 3.11+ is required; this workspace's existing interpreter is Python 3.13.15. All installation uses `uv`; no project package should be installed globally. Run commands from the project root.

Windows PowerShell:

```powershell
Test-Path .venv
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Linux, with an existing `.venv`:

```bash
test -d .venv
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install --python .venv -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Alternatively, activate the existing environment and run `uv run --active python main.py ...`.
There was no pre-existing `pyproject.toml` or `uv.lock`; `requirements.txt` is the install manifest. `requirements-lock.txt`, if present, records the tested environment and can be installed with the same `uv pip install --python .venv -r requirements-lock.txt` command. It is platform-specific; use the main requirements on Colab or another OS.

CPU works without a GPU. CUDA requires an NVIDIA GPU, compatible driver and a CUDA-enabled PyTorch build. Check with:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If you need a different PyTorch build, use the command selected by the official [PyTorch installer](https://pytorch.org/get-started/locally/), translating installation to `uv pip install --python .venv ...`. Do not replace the environment. Model weights download on first use; offline users must supply a previously obtained trusted `.pt` path with `--model`.

## Run the CLI

```powershell
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --model yolo26n.pt --tracker bytetrack.yaml --save
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --classes 0 2 3 5 7 --confidence 0.1 --device cpu --line 0.1 0.5 0.9 0.5 --counting
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --stride 2 --max-width 960 --imgsz 480 --no-show-trajectories
.venv\Scripts\python.exe main.py --source webcam:0 --max-frames 300
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --compare
.venv\Scripts\python.exe main.py --help
```

General-purpose examples (no class restriction):

```powershell
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --model yolo26s.pt --imgsz 640
.venv\Scripts\python.exe main.py --source data/input/evaluation/dog.jpg --model yolo26s.pt
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --tracker botsort.yaml --track-buffer 45
.venv\Scripts\python.exe main.py --source data/input/video.mp4 --min-track-hits 3 --class-window 15 --show-tentative
```

To compare models on your own inputs, run:

```powershell
.venv\Scripts\python.exe -m utils.evaluate_models --images data/input/bus.jpg data/input/evaluation/dog.jpg --video data/input/video.mp4 --sequence-frames 160
```

This saves per-frame JSON, CSV timings, annotated examples, sequence videos and label-transition diagnostics under `data/output/model_evaluation`. It uses all classes and equal thresholds for nano/small comparisons. Its counts and label transitions are not mAP, recall or ground-truth ID switches.

On Linux replace `.venv\Scripts\python.exe` with `.venv/bin/python`. Outputs are always saved; `--save` is accepted for compatibility. A local webcam CLI run without `--max-frames` continues until interrupted; set a frame limit to ensure finalized exports. The dashboard Stop button saves a partial run. It cannot interrupt a model download or an inference call already in progress.

COCO examples: person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7, cat 15, dog 16. The dashboard lists every class. The detector uses `nms=None`, selecting YOLO26's NMS path so the IoU control has a concrete effect. The low default confidence of 0.1 allows ByteTrack's low-confidence association stage to receive candidates. Increasing it can starve that stage.

## Dashboard demonstration

**All objects is the default** and enables all 80 COCO categories. Road vehicles remains optional; Custom classes lets you choose any subset. No prediction is forcibly converted into a vehicle.

The default model is **YOLO26s at 640**. In the saved spot-check it removed a duplicate cat prediction on a dog compared with nano; it did not fix the difficult annotated traffic clip. YOLO26n remains a faster CPU option. Larger models and 960/1280 inference are selectable, but more resolution can also add false positives. See [measured comparison and failure analysis](docs/evaluation_report.md).

**Tracker candidate confidence** defaults to 0.1. **Confirmation evidence / image confidence** defaults to 0.25: in video it defines strong evidence for a new track, not a per-frame cutoff. A video ID needs three observations within eight processed frames, at least two strong observations, and 60% weighted support for its current label before confirmation. A confirmed ID retains low-score observations during brief dips. If its last eight processed frames contain no strong evidence, confirmation is withdrawn until evidence returns. Confirmation is an evidence policy, not a correctness guarantee. Missing boxes are never synthesized. Tentative observations are hidden by default and never counted; enable the optional tentative overlay to inspect them. Still images use the threshold directly and have no invented IDs.

The last 15 processed frames supply confidence-weighted, exponentially decayed class votes. A competing label needs at least three observations and 1.25 times the incumbent's vote weight to replace it. Old votes expire. This resists isolated mistakes without locking an incorrect first label forever. It cannot repair a consistently wrong model or a physical ID switch. Disable class stabilization for comparison.

Near-identical simultaneous boxes from different IDs (IoU >= 0.95) are treated as duplicate observations. The guard prefers a confirmed, established identity; it does not merge or invent IDs or change raw classes. Suppressed observations have `duplicate_of` set and cannot increment confirmed counts. This can suppress a genuinely distinct heavily overlapping object: lower the limit cautiously or set `--duplicate-iou 0` to disable it. This is geometric duplicate control, not a solution for incorrect labels on distinct boxes.

The previous `--display-confidence` option is preserved, but now controls confirmation evidence (or image reporting) as described above. It no longer discards every low-confidence observation from an already confirmed track. Raw class, score, weighted support, confirmation state and duplicate provenance remain auditable in CSV.

1. Open the local Streamlit URL printed in the terminal.
2. Select Upload video or image and choose an input. Model loading occurs only when Start is pressed.
3. Keep All objects or choose classes; select inference size. Counting lines are off by default and can be enabled with normalized endpoints.
4. Start processing and watch the annotated preview. Sidebar changes affect only the next run.
5. Open Analytics and Tracking Data after completion; download MP4, CSV and JSON.
6. Inspect saved run metadata under Project Information.

The local webcam mode uses OpenCV on the **server's** device. It cannot access a remote visitor's webcam. For a remotely hosted demo, upload video. Camera processing is sampled as quickly as inference allows; timestamps use elapsed wall time, whereas playback uses the nominal camera FPS and can play faster than capture. Uploaded video is the recommended source for timing experiments.

The model is retained throughout a worker's run, never reloaded per frame. Mutable YOLO predictor/tracker objects are intentionally not globally cached across users. Simple COCO metadata is cached. A new run gets a fresh model; Streamlit reruns do not start another processing job. Multiple simultaneous users load separate models and may exhaust GPU memory. This is a local lab dashboard, not a multi-tenant production service.

## Outputs and meanings

Each run creates `data/output/<run-id>/`:

| File | Contents |
|---|---|
| `annotated.mp4` | H.264, browser-friendly, no source audio |
| `annotated_mpeg4.mp4` | Fallback when H.264 conversion fails; download and play externally |
| `tracking.csv` | Reported observations, stabilized and raw classes, raw confidence, vote support, confirmation, box, centroid, frame and seconds |
| `observations.csv` | All returned tracker observations before application confirmation filtering (not all internal YOLO candidates) |
| `tracker.yaml` | Exact tracker configuration used for that video run |
| `annotated.jpg` | Still-image detection output; IDs remain blank |
| `tracks.csv` | First/last observation and elapsed span per tracked ID |
| `counts.csv` | Unique IDs and A→B/B→A crossings per class; image runs use `image_detections` |
| `frames.csv` | Per-frame counts, detector/tracker time and processing FPS |
| `analytics.json` | Configuration, aggregate measured metrics, warnings and output metadata |

SQLite `data/output/history.db` retains run metadata and summaries. Uploaded temporary copies are deleted when their worker finishes; exported detections remain on disk. Failed runs may leave partial output directories for diagnosis. Delete only unwanted run folders manually when disk space is needed.

Coordinates refer to resized output pixels. Frame numbers are 1-based source indices. Video timestamps are `(frame_number - 1) / source_fps`. Trail memory is bounded per track, but full records are kept in memory until export; use short lab clips or `--max-frames` for large inputs.

**Unique objects** means unique tracker IDs by observed class, not verified physical individuals. Each ID belongs to exactly one current stabilized class; if its label changes, the unique count and previous crossing totals move to that class instead of double-counting the ID. **Visible objects** means the number of detections returned for that frame. **Crossings** require a centroid moving between opposite sides of the finite line outside a pixel dead band. For a left-to-right horizontal line, A is above and B below. Each ID can contribute at most one count per direction per run; genuine repeated trips in the same direction are intentionally not counted again. A new ID after an occlusion may still inflate counts. Last observation is not a confirmed physical exit.

Skipping frames reduces output FPS to `source_fps / stride`, preserving approximate video duration. It increases inter-frame motion and weakens association; tracker buffer values are measured in processed frames. Inference always receives original source pixels and Ultralytics performs its own aspect-preserving letterbox resize. `--max-width` affects only rendered output; 0 (the default) keeps source dimensions. Odd video edges are padded by one pixel for H.264 instead of stretching. Image output preserves odd dimensions. Increasing inference resolution cannot restore detail absent from the source. All video resources are released in `finally` blocks.

Processing FPS includes initialization and the frame loop, excluding final transcoding/exports. Per-frame FPS includes detection, annotation and writing; inference milliseconds measure the complete detector/tracker call. Neither is pure neural-network latency. Comparison mode uses separate state for each tracker; compare repeated runs under identical settings and consider warm-up effects. No mAP, MOTA or IDF1 is claimed without ground truth.

## Architecture and learning materials

`app.py` and `main.py` use `src/video_processor.py`. Detection parsing, track histories, counting, rendering, evaluation, exports and database access live in separate modules under `src/`. Defaults are in `config/settings.yaml`. See [architecture](docs/architecture.md), [algorithms](docs/algorithm.md), [lab report](docs/lab_report.md) and [viva answers](docs/viva_questions.md).

Open `notebooks/colab_demo.ipynb` in Colab and follow its cells. It installs into an existing Colab environment via `uv`, processes an uploaded video independently of Streamlit, and downloads output. The notebook does not create a virtual environment or modify this project's `.venv`.

## Tests

```powershell
.venv\Scripts\python.exe -m compileall -q src utils app.py main.py tests
.venv\Scripts\python.exe -m pytest -q
```

Tests use mock detections, real ByteTrack/BoT-SORT association on synthetic boxes, and generated videos; no GPU or downloaded model is required. They cover parsing IDs and boxes, empty results/frames, trails, resets, unique and directional counts, jitter, finite line endpoints, invalid input, exports, video decoding, fallback and cancellation. Actual validation results are recorded in `docs/validation.md`.

## Troubleshooting and limitations

- First model load fails: verify internet access or use a local weights path. Download only trusted model weights.
- CUDA unavailable: choose auto or CPU; installing the generic wheel does not guarantee CUDA support.
- Unsupported/corrupt video: re-encode with FFmpeg; a filename extension does not guarantee decoder availability. A decode ending early is reported as a partial-video warning when frame count metadata allows detection.
- No detections: use relevant COCO classes, reduce confidence, improve lighting, or increase resolution. Empty outputs still export valid CSV headers.
- Camera unavailable: close other camera apps, check OS permissions, and choose the correct device index.
- Browser video fails: download fallback MP4 and play in VLC. Convert with `ffmpeg -i annotated_mpeg4.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart annotated.mp4`.
- Permission errors: use a writable project/output directory and close locked output files. Set `UV_CACHE_DIR` as above when the default cache is inaccessible.
- Performance: reduce model size/inference resolution or frame count; output width mainly changes rendering/encoding cost; skipping trades tracking continuity for speed.
- IDs can switch under occlusion, fast motion or crowded scenes. ByteTrack has no appearance re-identification and does not guarantee persistent identity after long disappearance.
- No audio is retained; no ground-truth accuracy evaluation is implemented. Hardware-specific real-time behavior must be measured.

Ultralytics licensing applies to its models/library; review its AGPL/enterprise terms before redistributing or deploying beyond the lab. References: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [tracking API](https://docs.ultralytics.com/modes/track/), [ByteTrack paper](https://arxiv.org/abs/2110.06864).
