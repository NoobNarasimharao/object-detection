# General-purpose detection and tracking: implementation and evaluation

## Scope and preserved environment

All changes use the existing `.venv` and its installed dependencies. No environment was recreated and no global packages were installed. `uv pip check --python .venv` verified 67 compatible packages. Tests and inference used Python 3.13.15, Ultralytics 8.4.198 and PyTorch 2.15.0+cpu. CUDA was unavailable. Official YOLO26s weights were downloaded alongside the existing nano weights; both remain ignored by Git.

The repository review covered the UI, CLI, configuration, detector, native tracking integration, histories, counting, rendering, exports, database, tests, documentation and standalone notebook. Existing video exports, SQLite history, cancellation, analytics, webcam path and optional vehicle filtering were retained.

## Findings and changes

1. The earlier per-frame reporting filter removed low-confidence tracker observations from downstream histories. It has been replaced with temporal confirmation: three observations within eight processed frames, two sufficiently strong observations, and weighted class support. Brief low-score recovery observations are retained after confirmation; persistent weak evidence revokes confirmation. This reduces unstable reporting but can miss very short-lived objects.
2. The model's latest class could replace an object's label immediately. A bounded 15-frame confidence-weighted vote now resists isolated mistakes and allows sustained new evidence to replace an old label. Raw class and raw score remain exported. This is temporal inference, not a semantic rule such as “phone means car.”
3. One ID could inflate multiple class counts. Counts and prior crossing events now migrate to the ID's current stabilized class. Historical per-frame labels remain unchanged in the audit records.
4. The saved output width previously limited the pixels supplied to inference. Inference now sees original source frames; output resizing happens afterward. Odd video dimensions are padded rather than stretched. Still-image prediction is supported without artificial tracking IDs.
5. The sequence review found nearly identical simultaneous boxes with distinct IDs. A configurable 0.95-IoU guard now prefers the established confirmed track and records suppressed observations in `duplicate_of`. It does not merge identities. A genuinely separate heavily overlapping object may be suppressed; the guard can be disabled.
6. All 80 COCO classes are enabled by default. Road vehicles is optional, and arbitrary class subsets remain available. YOLO26s at 640 is the balanced default; nano remains the faster choice. Counting lines default off; class colors, count overlays, trajectories and tentative overlays are independently configurable.

## Comparison method

Three clean public reference images were used: the Ultralytics bus image (people and a bus), the PyTorch Hub dog image (one dog), and the Ultralytics two-person image (people and ties). Five frames from the supplied traffic output were also sampled at zero-based frame indices 30, 60, 120, 200 and 380.

**The traffic input is already annotated.** It is 480×270 at 50 FPS, with 536 frames. Burned-in boxes, text and the yellow counting arrow contaminate detection. For example, nano produced a kite box around the arrowhead. Its results are diagnostic examples, not a clean traffic benchmark. The unusual top-down views, partial vehicles at image borders and limited detail are plausible additional causes of classification ambiguity; their individual contributions have not been isolated.

The comparison uses YOLO26n at 640, YOLO26s at 640, and YOLO26s at 960. All classes are enabled in every variant. Candidate confidence is 0.1, reporting threshold for the image comparison is 0.25, and NMS IoU is 0.5. One warm-up call precedes three timed calls per sample. Timing includes preprocessing and postprocessing but excludes model loading. No class filtering was used to make one model look better.

| Variant | Mean call time across 8 samples (CPU) |
|---|---:|
| YOLO26n, 640 | 55.14 ms |
| YOLO26s, 640 | 114.50 ms |
| YOLO26s, 960 | 258.35 ms |

These are local spot measurements, not universal benchmarks or a guarantee of real-time throughput. Sources, shapes, full predictions and per-sample timing are in [predictions.json](../data/output/model_evaluation/predictions.json) and [comparison.csv](../data/output/model_evaluation/comparison.csv).

## Visual observations: positives and failures

| Example | Nano 640 | Small 640 | Small 960 |
|---|---|---|---|
| Bus image | 1 bus, 4 people | 1 bus, 4 people | 1 bus, 4 people |
| One-dog image | Dog 0.694 plus an erroneous overlapping cat 0.378 | One dog 0.921; no cat above threshold | One dog 0.828; visibly tighter/incomplete horizontal box |
| Two-person image | 2 people, 1 tie | 2 people, 1 tie | 3 person boxes, including a duplicate, plus 1 tie |
| Traffic frame 60 | Cell phone on gray vehicle; kite on arrowhead | No boxes above comparison threshold despite visible vehicles | Person prediction on the partially visible red vehicle |
| Traffic frame 200 | Phone/kite/truck/car predictions | Bus on red car and traffic-light prediction | Bus on red car |

The dog case supports a specific observed improvement, not a general accuracy claim. Higher resolution did not reliably improve these examples. The traffic frames demonstrate both false positives and missed vehicles; the larger model did not fix this input universally. The default choice of small/640 balances the clean-image result against CPU cost and the additional failure observed at 960.

Review examples:

- [Nano dog/cat duplicate](../data/output/model_evaluation/dog_yolo26n_640.jpg)
- [Small 640 dog result](../data/output/model_evaluation/dog_yolo26s_640.jpg)
- [Small 960 duplicate person](../data/output/model_evaluation/zidane_yolo26s_960.jpg)
- [Traffic misclassifications remain](../data/output/model_evaluation/video_0200_yolo26s_640.jpg)

## Tracking review

Both models were run with ByteTrack over the first 160 consecutive frames of the supplied annotated clip, without skipping, class filtering or fabricated detections. The final runs use the current confirmation, class-history and duplicate-guard settings.

| Diagnostic | Nano 640 | Small 640 |
|---|---:|---:|
| Observed IDs (including tentative) | 5 | 4 |
| Raw class transitions within IDs | 22 | 28 |
| Stabilized class transitions within IDs | 4 | 5 |
| Suppressed near-identical observations | 0 | 10 |
| Total confirmed unique IDs | 5 | 2 |
| Processing FPS, including initialization | 12.24 | 8.68 |

Fewer label transitions are an expected effect of smoothing; they do **not** establish correctness. The final small-model count was two car IDs, but intermediate labels included person, suitcase and cell phone. The nano result retained erroneous phone/kite labels and fragmented the gray vehicle's identity. In the pre-guard small-model run, the gray vehicle had simultaneous IDs 1 and 8 at source frame 141; the final guard suppresses the near-identical newcomer while retaining its raw observation. This is duplicate suppression, not proof of successful re-identification across disappearance.

The separate real-tracker tests use synthetic moving boxes with a two-frame gap. Both ByteTrack and BoT-SORT maintained the same ID in those controlled tests. Their success does not establish identity stability in crowded or occluded real footage.

No full ground-truth trajectory annotation exists for this clip. Consequently, true ID-switch counts, MOTA, IDF1, mAP, precision and recall are **not reported**. Label changes, observed IDs and suppressed duplicates are not substitutes for those metrics.

- [Sequence diagnostics and exact output paths](../data/output/model_evaluation/sequence_comparison.json)
- [Pre-guard review showing duplicate IDs](../data/output/model_evaluation/sequence_review.jpg)
- [Final sequence review](../data/output/model_evaluation/sequence_review_final.jpg)

## Validation

- **40 automated tests passed in 7.84 seconds.** Coverage includes all-class and multi-class configuration, confirmation, class-vote recovery, low-confidence continuity, duplicate provenance, counts after reclassification, real ByteTrack/BoT-SORT short-gap association, image outputs without fake IDs, original-resolution inference, even-edge padding, CSV/JSON/video exports, and image/video dashboard rendering.
- Real CLI image detection with YOLO26s produced a dog detection and an annotated JPEG at the source dimensions, with no track ID.
- Real CLI BoT-SORT processing of the existing six-frame reference clip completed with one bus and four person identities and MP4/CSV/JSON exports.
- Updated Streamlit server returned HTTP 200; the temporary validation server was stopped afterward.
- Dependency compatibility, Python source syntax and notebook code-cell syntax were checked using the existing interpreter.

## Remaining limits and next evidence needed

The clean examples cover only a small subset of COCO classes; they do not establish performance across all 80 categories. A clean original traffic clip and representative labeled scenes are needed for quantitative accuracy and identity evaluation. Low-resolution top-down footage may require domain-specific training data. Temporal stabilization cannot correct a persistently wrong classifier or guarantee recovery after an ID switch. Confirmation can suppress brief appearances; overlap suppression can suppress genuinely distinct near-coincident boxes. Use the audit CSV and configurable controls when investigating either case.

CUDA execution and a physical webcam were not tested. Colab was syntax checked, not executed remotely; its standalone notebook remains a simpler teaching example. Long videos still accumulate records in memory. None of these limitations was hidden by reporting a higher confidence threshold or a vehicle-only default.

## Reproduce

From the project root, using the existing environment:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m utils.evaluate_models --images data/input/bus.jpg data/input/evaluation/dog.jpg data/input/evaluation/zidane.jpg --video path/to/original.mp4 --sequence-frames 160
```

Public reference sources: [Ultralytics bus](https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg), [Ultralytics two-person image](https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg), [PyTorch Hub dog](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg). API references: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) and [YOLO26 inference](https://docs.ultralytics.com/models/yolo26/).
