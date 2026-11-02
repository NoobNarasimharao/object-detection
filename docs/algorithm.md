# Detection and association

YOLO predicts class scores and bounding boxes in a single forward pass. A convolutional backbone extracts spatial features; multi-scale feature aggregation supports different object sizes; detection heads predict locations and classes. YOLO26 supports an end-to-end NMS-free path and a conventional NMS path. This project explicitly selects `nms=None` so the configured IoU threshold controls NMS. It does not describe YOLO26 as the original fixed-grid YOLOv1 architecture. Pretrained COCO weights support 80 categories without custom training.

For boxes A and B, `IoU(A,B) = area(A ∩ B) / area(A ∪ B)`. NMS suppresses overlapping lower-score candidates of a class. Confidence is the detector's score, not a calibrated probability of correctness or a measure of tracking accuracy.

ByteTrack uses a Kalman filter to predict track motion and two detection association stages. High-score detections match existing predicted tracks first. Remaining eligible tracks then match lower-score detections, helping recover objects whose confidence dropped under occlusion. Unmatched high-score candidates may initialize new tracks; lost tracks remain available for a configured buffer before removal. Linear assignment minimizes a matching cost based on overlap (with implementation-specific gating/score fusion). The Hungarian algorithm is the standard conceptual explanation; Ultralytics may use a Jonker–Volgenant solver via LAP for the linear assignment problem.

Kalman prediction is `x̂(t|t−1) = F x̂(t−1|t−1)` and `P(t|t−1) = F P Fᵀ + Q`. Measurement update uses `K = P Hᵀ (H P Hᵀ + R)⁻¹` and `x̂(t|t) = x̂(t|t−1) + K(z − Hx̂(t|t−1))`. Here F models motion, H maps state to measurements, and Q/R describe process/measurement uncertainty. Tracking state includes box geometry and velocity; no manual IDs are invented by this application.

## Counting and trajectories

Centroid: `c = ((x1+x2)/2, (y1+y2)/2)`.
For line start a and end b, signed distance is `d = ((bx−ax)(cy−ay) − (by−ay)(cx−ax)) / ||b−a||`. Negative is side A; positive is B. Observations within a tolerance band do not change the last stable side. On a stable side transition, the movement segment must intersect the finite counting segment. A set of `(track_id, direction)` keys suppresses repeat counts. This explicitly trades repeat-trip counting for resistance to jitter.

Unique counts assign each confirmed tracker ID to its current stabilized class. Reclassification migrates the ID and its crossing events rather than duplicating them. Trajectories are fixed-length deques per ID. Full positions remain in tracking CSV. Duration is the span between first and last observation, including gaps; it is not the sum of continuously visible intervals. Neither a disappearance nor the last video frame proves physical exit.

## Evaluation and limits

`FPS = processed frames / elapsed processing seconds`. Mean detector/tracker latency averages wall-clock call durations; confidence mean/min/max summarize returned scores. These are operational metrics, not accuracy. MOTA requires ground-truth trajectories and false-negative, false-positive and identity-switch counts; IDF1 requires identity matching; mAP requires labeled boxes. This project deliberately does not infer them from confidence.

BoT-SORT is available as a comparison using its packaged YAML; it adds camera-motion compensation and supports optional appearance re-identification. This application does not enable custom ReID settings. ByteTrack remains primary. Compare the same input/settings and include warm-up and hardware in observations. Occlusion, camera motion, class changes, small objects and frame skipping can change IDs or counts.

References: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [YOLO26](https://docs.ultralytics.com/models/yolo26/), [ByteTrack](https://arxiv.org/abs/2110.06864), [BoT-SORT](https://arxiv.org/abs/2206.14651).

## Temporal confirmation and class consistency

`TrackStabilizer` consumes all returned tracker observations before reporting. Three distinct processed-frame observations within eight frames, two scores at least 0.25, and at least 60% weighted class support confirm a track. Low-score recovery boxes remain usable for confirmed tracks, but eight frames without strong evidence revoke confirmation. No boxes are fabricated during missing frames. Lost state expires after the configured buffer. Tentative tracks never enter counting or confirmed histories.

For class k, `vote(k,t) = sum(conf_i * 0.9**(t-i))` over the last 15 processed frames for that ID. A challenger requires three observations and a vote greater than 1.25 times the incumbent's. This bounded, decaying history can correct an initially wrong class. The exported detector confidence belongs to `raw_class`; `class_support` is the vote fraction for `object_class`, not a calibrated probability. Identity switches remain a failure mode: a shared ID can mix evidence from different physical objects.

All 80 classes are enabled by default. Original pixels enter inference, and optional output resizing maps boxes afterward. Still images use `predict`, export no tracking IDs, and have separate detection counts. A larger model is an option, not a proof of accuracy. See `evaluation_report.md` for actual comparisons.

Near-identical boxes from different IDs are suppressed at IoU >= 0.95 by default, preferring confirmed/older tracks. Suppression is recorded in `duplicate_of`; the suppressed raw identity and class are retained. No IDs are merged. Disable with `--duplicate-iou 0` when very close real objects are being suppressed. Unique-count deduplication across non-overlapping times is not attempted without identity evidence.
