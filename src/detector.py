from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    raw_label: str | None = None
    raw_class_id: int | None = None
    class_support: float = 1.0
    confirmed: bool = True
    duplicate_of: int | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

def extract_detections(result, width: int, height: int) -> list[Detection]:
    """Read tracker-owned IDs, preserving untracked detections as None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
    output = []
    for box, cls, conf, tid in zip(xyxy, boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), ids):
        if not np.isfinite(box).all() or not np.isfinite(conf) or not np.isfinite(cls):
            continue
        x1, y1, x2, y2 = box
        box = (float(np.clip(x1, 0, width - 1)), float(np.clip(y1, 0, height - 1)),
               float(np.clip(x2, 0, width - 1)), float(np.clip(y2, 0, height - 1)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        valid_id = int(tid) if tid is not None and np.isfinite(tid) else None
        output.append(Detection(valid_id, int(cls), result.names[int(cls)], float(conf), box, confirmed=False))
    return output

class YOLODetector:
    """One model instance per processing job: no shared mutable tracker state."""
    def __init__(self, config):
        import torch
        from ultralytics import YOLO
        self.config = config
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
        if self.device != 'cpu' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Select CPU or auto.')
        try:
            self.model = YOLO(config.model)
        except Exception as exc:
            raise RuntimeError('Cannot load model weights. Check the model path or internet access for the first download.') from exc

    def reset(self):
        if self.model.predictor is not None:
            for tracker in getattr(self.model.predictor, 'trackers', []):
                tracker.reset()
            self.model.predictor = None

    def configure_tracking(self, folder):
        """Save the exact tracker settings with the run for reproducibility."""
        import yaml
        from ultralytics.utils import ROOT
        settings = yaml.safe_load((ROOT / 'cfg/trackers' / self.config.tracker).read_text())
        settings.update(track_buffer=self.config.track_buffer, match_thresh=self.config.match_threshold)
        self.tracker_path = str(folder / 'tracker.yaml')
        (folder / 'tracker.yaml').write_text(yaml.safe_dump(settings), encoding='utf-8')

    def predict(self, frame):
        """Single images have detections, not temporal identities."""
        if frame is None or frame.size == 0:
            raise ValueError('Cannot process an empty frame.')
        c = self.config
        result = self.model.predict(frame, conf=c.confidence, iou=c.iou, imgsz=c.imgsz,
                                    device=self.device, classes=c.classes or None, nms=None, verbose=False)[0]
        return extract_detections(result, frame.shape[1], frame.shape[0])

    def track(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError('Cannot process an empty frame.')
        c = self.config
        result = self.model.track(frame, persist=True, tracker=getattr(self, 'tracker_path', c.tracker), conf=c.confidence,
                                  iou=c.iou, imgsz=c.imgsz, device=self.device,
                                  classes=c.classes or None, nms=None, verbose=False)[0]
        # Confirmation and label history must see low-score tracker observations too.
        return extract_detections(result, frame.shape[1], frame.shape[0])
