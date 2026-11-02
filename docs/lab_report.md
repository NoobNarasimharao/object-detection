# Real-Time Multi-Object Detection and Tracking Using YOLO and ByteTrack

## 1. Experiment title
Real-Time Multi-Object Detection and Tracking Using YOLO and ByteTrack.

## 2. Aim
Implement and study a complete video analysis pipeline that detects multiple objects, associates their identities over time, visualizes trajectories, and exports reproducible operational measurements.

## 3. Problem statement
An image detector locates objects independently in each frame. Video applications also need temporal identity, movement and event counts. Build a system that connects detection to tracking and explains where identity and counting can fail, while supporting a practical laboratory demonstration.

## 4. Objectives
Detect pretrained COCO categories; extract tracker-generated IDs; maintain histories; distinguish simultaneous, unique-ID and line-crossing counts; process files and a local camera; export annotated video and tabular observations; measure processing performance without fabricating accuracy.

## 5. Software and hardware requirements
Python 3.11+, the existing `.venv`, uv, Ultralytics, PyTorch, OpenCV, NumPy, Pandas, Streamlit, Plotly, SQLite, PyYAML, LAP and imageio-ffmpeg. A CPU suffices for functional testing. An NVIDIA GPU with a compatible CUDA-enabled PyTorch build may improve throughput. A local camera is optional. RAM/storage requirements depend on video duration, resolution and detection density; start with a short clip. Internet is needed for initial dependency and weight downloads.

## 6. Introduction
Object classification assigns a label to an image or region; detection also localizes each instance. Multi-object tracking links observations across time. Detection and tracking are complementary: detection supplies semantic/spatial evidence, and tracking uses temporal continuity. Useful applications include pedestrian flow, road traffic and animal monitoring. A tracker ID is an algorithmic hypothesis, not biometric identity.

## 7. YOLO architecture
YOLO is a family of single-stage detectors. Convolutional layers learn spatial features from pixels, early layers emphasizing local patterns and deeper layers encoding semantic structure. Multi-scale feature aggregation allows the detection head to work with different object sizes. Heads produce bounding-box geometry and class scores. Pretraining on COCO provides useful representations and 80 supported classes without collecting a custom training set.

YOLO26 has lightweight DFL-free box regression and supports native end-to-end inference. This implementation uses its NMS path with `nms=None`, retaining a meaningful configurable NMS IoU threshold. The original YOLOv1 grid formulation should not be assumed to describe every modern head. NMS removes redundant overlapping candidates, while confidence filtering limits which detections reach tracking.

## 8. ByteTrack working principle
ByteTrack predicts existing track locations using a Kalman motion model. It first associates high-confidence detections with predicted tracks using an overlap-based assignment cost. It then uses lower-confidence detections to rescue eligible unmatched tracks. This second stage can recover partially occluded objects that a high confidence cutoff would otherwise discard. High-confidence unmatched detections can start new tracks. Tracks can remain lost for a buffer period before removal.

The data association is a linear assignment problem. The Hungarian algorithm is a standard solution; the installed Ultralytics implementation can use a LAP/Jonker–Volgenant solver. ByteTrack does not require appearance embeddings. Strong occlusion, abrupt motion and similar overlapping objects can still cause ID switches.

## 9. Object detection methodology
Decode a frame with OpenCV and pass the original pixels to the model. Resize only the rendered output afterward, mapping the returned coordinates consistently. Ultralytics performs inference preprocessing and maps predictions back to the supplied frame coordinates. Apply the selected class filter, confidence threshold and NMS setting. Extract finite, valid boxes, clip to output bounds and preserve confidence/class values. Handle absent or empty results as an empty observation set.

## 10. Multi-object tracking methodology
Call `model.track(frame, persist=True, tracker="bytetrack.yaml", ...)` on consecutive processed frames. Extract IDs from `result.boxes.id`; do not manufacture IDs for unmatched detections. Store centroid observations and bounded trails keyed by IDs. Initialize fresh model/tracker state for independent runs. Record first and last observations, visible detections and unique ID sets. Missing IDs remain blank in exports and do not enter tracked-object counters.

## 11. System architecture
Input → OpenCV frame decoding → optional frame sampling/resizing → YOLO26 → ByteTrack → ID extraction → histories/counters → OpenCV annotation → video/CSV/JSON/SQLite → Streamlit/Plotly dashboard. The CLI and dashboard share the same engine. The background worker publishes a lock-protected progress snapshot; cancellation is checked between frames. Capture and writer cleanup occur in a `finally` block.

## 12. Mathematical formulas

- Intersection over Union: `IoU = |A ∩ B| / |A ∪ B|`.
- Centroid: `cx = (x1 + x2)/2`, `cy = (y1 + y2)/2`.
- Kalman state prediction: `x⁻ = F x`; covariance prediction: `P⁻ = F P Fᵀ + Q`.
- Kalman gain: `K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹`.
- State update: `x = x⁻ + K(z − Hx⁻)`; covariance update: `P = (I − KH)P⁻`.
- Assignment: minimize `Σᵢⱼ Cᵢⱼ Xᵢⱼ`, where binary X permits at most one selected detection for each track and vice versa; a basic overlap cost is `Cᵢⱼ = 1 − IoUᵢⱼ` with gating.
- Signed line distance: `d = ((bx−ax)(cy−ay) − (by−ay)(cx−ax))/sqrt((bx−ax)²+(by−ay)²)`.
- Mean latency: `t̄ = (Σ tᵢ)/N`; processing rate: `FPS = N/T`.
- Unique class count: cardinality of `{track IDs observed with class k}`.
- Observation span: `last_seen_time − entry_time`.

F models motion, H maps state to observed box geometry, Q is process noise, R is measurement noise, and z is the measurement. The counting algorithm ignores the dead band, checks finite-segment intersection and stores already-counted ID/direction pairs.

## 13. Step-by-step algorithm
1. Validate input, configuration and output path.
2. Open capture, read source FPS/frame count and prepare fresh state.
3. Load pretrained YOLO and choose CUDA or CPU.
4. Read a frame, enforce optional stride, resize without material distortion.
5. Run YOLO and persistent ByteTrack association.
6. Validate returned boxes and extract IDs, labels and confidence.
7. Update per-ID trajectories, unique sets and crossing state.
8. Draw boxes, labels, centroids, trails and counting line.
9. Write video and append records/timing statistics; publish progress.
10. Repeat until EOF, configured limit or cancellation.
11. Release resources, convert to H.264 when possible, and export structured results.
12. Store history and inspect charts/downloads.

## 14. Implementation details
`ProcessingConfig` validates settings. `YOLODetector` owns the model; `extract_detections` handles result parsing. `TrackHistory` maintains bounded visual histories. `ObjectCounter` uses a signed-distance dead band and one event per ID per direction. `VideoProcessor` coordinates IO, inference and exports. `Database` uses parameterized SQLite statements. Workers are isolated to avoid sharing mutable tracking state between users. Each run has a UUID folder to avoid collisions. Uploaded temporary files are removed after processing.

Reported observations are exported through `tracking.csv`; all returned tracker observations before application confirmation are in `observations.csv`; `tracks.csv` summarizes first/last observations. Output video drops audio. H.264 conversion uses bundled FFmpeg if available, otherwise the MPEG-4 fallback remains downloadable. Output FPS is adjusted for frame stride. Decoder support, rather than filename extension alone, determines whether a file opens.

## 15. Experimental procedure
Use the existing environment and follow README installation. Record hardware, OS, Python/PyTorch/Ultralytics versions, model, resolution, thresholds, classes and stride. Select a short clip with visible motion across a line. Run ByteTrack, inspect ID continuity, compare unique counts with simultaneous counts, and export results. Repeat with a more difficult clip or higher stride. Optionally run CLI `--compare` to process the identical video with BoT-SORT. Use several repetitions; separate warm-up effects and do not treat one run as a universal benchmark.

For manual validation, inspect selected frame numbers and compare rendered centroids with CSV coordinates. Check that a crossing occurs within the finite segment and outside the dead band. A generated blank test clip validates software output, not real-world detection quality.

## 16. Results — fill from actual experiments

The following table intentionally contains unmeasured placeholders, not invented observations. Software test outcomes are maintained separately in `validation.md`.

| Measurement | Actual value |
|---|---|
| Input video and duration | [record after experiment] |
| CPU/GPU and software versions | [record after experiment] |
| Model/tracker/settings | [record after experiment] |
| Processed frames | [from analytics.json] |
| Processing seconds and FPS | [from analytics.json] |
| Average detector/tracker call time | [from analytics.json] |
| Unique IDs per class | [from counts.csv] |
| A→B and B→A crossings | [from counts.csv] |
| Observed failure cases | [manual inspection] |

Attach a representative annotated frame and screenshots of analytics after performing the experiment. Do not label detector confidence as tracking accuracy.

## 17. Performance evaluation
The implemented evaluator reports processed frame count, total processing time, overall FPS, mean detector/tracker call time, and confidence mean/min/max. Processing time includes model initialization and the loop but excludes final transcoding/exports. Per-frame timing excludes initialization. Compare like-for-like definitions and settings.

Precision/recall and mAP need labeled boxes; MOTA, IDF1 and identity switches need ground-truth trajectories and an appropriate matching protocol. These optional accuracy metrics are not implemented or asserted here. Confidence distributions characterize model scores only. If a labeled dataset is later added, use an established evaluator and document matching thresholds/splits.

## 18. Advantages
Pretrained models enable immediate demonstrations. ByteTrack is computationally economical and uses low-confidence evidence. Shared core logic keeps CLI/UI behavior consistent. Exports make observations inspectable. Explicit counter semantics make lab discussion reproducible. CPU fallback permits testing without dedicated GPU hardware.

## 19. Limitations
COCO classes do not cover arbitrary domains. Small/distant objects, poor lighting and motion blur can reduce detections. IDs can change under occlusion, class predictions can fluctuate, and unique-ID counts can overestimate people or vehicles. Frame skipping harms association. Crossing logic deliberately limits each identity to one event per direction. Last seen is not confirmed exit. Long videos accumulate records in memory. Webcam timestamps and nominal playback rate may differ. Remote browser cameras are not supported by server-side OpenCV. No accuracy claim is possible without ground truth.

## 20. Applications
Laboratory study, pedestrian flow demonstrations, traffic analysis prototypes, animal movement visualization, sports motion review and industrial object flow experiments. Deployment requires domain validation, appropriate licensing and a retention design suited to the actual setting.

## 21. Conclusion
The application integrates detection, temporal association, visualization, counting and measured operational analytics in a reusable pipeline. Experimental conclusions about speed or reliability must be completed using actual recordings and hardware. It demonstrates both the benefits of low-confidence association and the limits of tracker-derived identity.

## 22. References
1. [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).
2. [Ultralytics tracking API](https://docs.ultralytics.com/modes/track/).
3. Zhang et al., [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), ECCV 2022.
4. Aharon et al., [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651), 2022.
5. [OpenCV documentation](https://docs.opencv.org/4.x/).
6. [PyTorch documentation](https://pytorch.org/docs/stable/index.html).

## General-purpose pipeline revision

The balanced default is now YOLO26s at 640 with all COCO classes enabled. Temporal confirmation and bounded, decaying class votes reduce unstable reporting; original/raw predictions are preserved for audit. Class corrections do not inflate unique IDs. Still images are supported without fabricated track IDs. Counting overlays are optional and off by default. See `evaluation_report.md` for measured model comparisons and `algorithm.md` for exact confirmation/voting rules. The standalone Colab notebook remains an introductory direct-ByteTrack example rather than a duplicate of the full application.
