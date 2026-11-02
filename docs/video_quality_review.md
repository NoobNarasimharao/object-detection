> Historical review of the earlier threshold-only change. The current general-purpose pipeline and measured comparison are documented in `evaluation_report.md` and README.

# Traffic clip review

The supplied `faeb0f6dca75504f01f6d77340c89fce.mp4` is an already annotated 480×270, 50 FPS, 536-frame output. Sampled frames show top-down views of vehicles entering partially through image borders.

The corresponding saved run `4e4ca0b57b01451d89881ea19f1c97ab` used YOLO26n, 640 inference size, all classes and confidence 0.1. Its records contain 380 cell-phone observations with median confidence 0.739 and maximum 0.942. Early partial-object detections alternate among person/car/bottle/phone at low confidence. Therefore raising confidence alone cannot eliminate all erroneous phone predictions. ByteTrack associates detections; it does not repair semantic classification.

Changes: explicit Road vehicles scene preset; separate default 0.35 reporting threshold applied after ByteTrack while keeping the 0.1 candidate threshold; matching CLI option and documented counting consequences. No automatic relabeling was introduced. Filtering irrelevant categories can leave missed vehicles and does not establish better recall.

Validation: 20 automated tests passed in 4.21 seconds, including a regression that ensures low-confidence candidates still reach tracking while weak outputs are omitted. No accuracy improvement is claimed on this clip: the supplied file has burned-in overlays, and the original temporary upload was removed after the earlier run. A clean model/resolution comparison requires the original unannotated source.
