from types import SimpleNamespace
import numpy as np
from src.detector import YOLODetector
from src.video_processor import ProcessingConfig

def test_tracking_arguments_explicit_and_persistent():
    calls = []
    def track(frame, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=None)]
    detector = YOLODetector.__new__(YOLODetector)
    detector.config = ProcessingConfig()
    detector.device = 'cpu'
    detector.model = SimpleNamespace(track=track)
    assert detector.track(np.zeros((48,64,3), dtype=np.uint8)) == []
    assert calls[0]['tracker'] == 'bytetrack.yaml'
    assert calls[0]['persist'] is True
    assert calls[0]['nms'] is None

def test_reporting_threshold_does_not_starve_tracker(monkeypatch):
    from src.detector import Detection
    calls = []
    detector = YOLODetector.__new__(YOLODetector)
    detector.config = ProcessingConfig(confidence=.1, display_confidence=.35, classes=[2])
    detector.device = 'cpu'
    def track(frame, **kwargs):
        calls.append(kwargs)
        return [object()]
    detector.model = SimpleNamespace(track=track)
    monkeypatch.setattr('src.detector.extract_detections', lambda *args: [
        Detection(1, 2, 'car', .15, (0, 0, 10, 10)),
        Detection(2, 2, 'car', .8, (10, 10, 20, 20))])
    results = detector.track(np.zeros((48, 64, 3), dtype=np.uint8))
    assert [d.track_id for d in results] == [1, 2]
    assert calls[0]['conf'] == .1
    assert calls[0]['classes'] == [2]

def test_default_all_classes_and_selected_multiclass(monkeypatch):
    calls = []
    detector = YOLODetector.__new__(YOLODetector)
    detector.device = 'cpu'
    def predict(frame, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=None)]
    detector.model = SimpleNamespace(predict=predict)
    for selected in [None, [], [0, 2, 16, 67]]:
        detector.config = ProcessingConfig(classes=selected)
        detector.predict(np.zeros((20, 30, 3), dtype=np.uint8))
        assert calls[-1]['classes'] == (selected or None)
