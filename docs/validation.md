# Current general-purpose revision validation

The expanded suite passed **40 tests in 7.84 seconds** using the original `.venv`. Both real tracker backends were exercised on controlled boxes and real CLI inference was run with YOLO26s. The updated image/video dashboard passed AppTest checks, and its local HTTP health endpoint returned 200. The temporary server was stopped. `uv pip check --python .venv` reported all 67 installed packages compatible; no new project dependencies or replacement environment were needed.

The complete measured model comparison, visual failures, sequence diagnostics, and limitations are in [evaluation_report.md](evaluation_report.md). GPU and physical-webcam operation remain untested; no ground-truth accuracy metrics are claimed. The records below describe the initial implementation and are retained as historical evidence.

# Initial implementation validation record

Validated on 2 November 2026 using the pre-existing `E:\CV-UNIT-4\.venv` (Python 3.13.15). No environment was created, replaced or reset. Dependencies were installed only with `uv pip install --python .venv -r requirements.txt`. The tested package snapshot is `requirements-lock.txt`.

Environment: Ultralytics 8.4.198, PyTorch 2.15.0+cpu, OpenCV 5.0.0, Streamlit 1.66.1. `torch.cuda.is_available()` returned false. `uv pip check --python .venv` checked 67 packages and reported all compatible.

| Check | Actual outcome |
|---|---|
| Source compileall | Passed |
| Colab code-cell AST syntax validation | Passed; notebook not executed on Colab |
| Automated pytest suite | **19 passed in 4.02 seconds** |
| CLI help | Passed |
| CLI comparison mode | Passed: independent ByteTrack and BoT-SORT runs, three frames each, video and exports generated |
| Dashboard AppTest with no model download | Passed in suite |
| Dashboard AppTest populated with real exported results | Passed |
| Streamlit local server health endpoint | HTTP 200; validation server stopped afterward |
| YOLO26n real inference + explicit ByteTrack | Passed on six repeated reference frames |
| Tracking ID extraction and persistence | Five IDs (one bus, four persons); IDs persisted over multiple frames |
| H.264 output | Created and decoded; six output frames |
| CSV/JSON and SQLite | Generated, parsed and checked |
| Empty detections | Passed in suite; valid header-only tracking CSV |
| Counting, line directions, dead band, finite segment and repeats | Passed in suite |
| Tracker-state reset, bounded trajectories, clipped boxes and centroids | Passed in suite |
| FFmpeg fallback and cancellation before frames | Passed in suite |

The first test attempt exposed a test-harness relative-path issue, which was fixed, and restricted Windows temporary-directory permissions. The successful full test run used the same existing interpreter with normal temporary-directory access. One sandbox rerun stalled and was stopped; it is not counted as a passing run.

The installed current Ultralytics API deprecates `end2end=False` in favor of `nms=None` for external NMS. Source, tests, docs and notebook were updated to the current API; the successful final inference check used `nms=None` and explicit `tracker='bytetrack.yaml'`.

## Real inference fixture

`utils/smoke_inference.py` downloaded the public Ultralytics bus reference image, made a six-frame MJPEG input, loaded official `yolo26n.pt` weights, and ran CPU inference at 320 pixels. The final run is `data/output/91471acd3daa454cb5f74ef8521892db/`. Its measured processing time was 3.171 seconds; initialization-inclusive processing FPS was 1.892; average detector/tracker call time was 262.268 ms. These values describe only this tiny integration fixture and are **not a real-time or accuracy benchmark**. The earlier cold run included network download and is not comparable. The stationary fixture correctly produced zero crossings; crossing behavior is verified using controlled geometric tests.

## Not validated here

- CUDA execution (installed PyTorch build is CPU-only).
- Physical webcam capture (no camera was exercised).
- Full Colab execution (syntax checked only).
- Real moving-video tracking quality, occlusion robustness, or accuracy against ground truth.
- All alternate YOLO model sizes and every platform-specific video decoder.

Use a real moving clip for the laboratory experiment and fill the explicitly marked measurement placeholders in `lab_report.md`. No mAP, MOTA, IDF1 or tracking-accuracy values were fabricated.
