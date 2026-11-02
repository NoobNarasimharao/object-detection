from types import SimpleNamespace
import numpy as np
import pytest
from src.detector import extract_detections, YOLODetector
from src.tracker import TrackHistory
from tests.test_counter import det

class Tensor:
    def __init__(self, values): self.values = np.asarray(values)
    def cpu(self): return self
    def numpy(self): return self.values

class Boxes:
    xyxy = Tensor([[-5, -3, 20, 30], [10, 10, 5, 5]])
    cls = Tensor([0, 0])
    conf = Tensor([.9, .8])
    id = Tensor([7, 8])
    def __len__(self): return 2

def test_extraction_clips_and_preserves_id():
    result = SimpleNamespace(boxes=Boxes(), names={0:'person'})
    found = extract_detections(result, 100, 100)
    assert len(found) == 1 and found[0].track_id == 7
    assert found[0].centroid == (10, 15)
    result.boxes.id = None
    assert extract_detections(result, 100, 100)[0].track_id is None

def test_empty_results():
    assert extract_detections(SimpleNamespace(boxes=None), 10, 10) == []

def test_history_reset_and_length():
    h = TrackHistory(2)
    for i in range(4): h.update([det(i, i)], i, i / 10)
    assert len(h.trails[1]) == 2
    assert h.summaries()[0]['duration_seconds'] == .3
    h.reset()
    assert not h.trails and not h.tracks

def test_model_state_reset():
    calls = []
    detector = YOLODetector.__new__(YOLODetector)
    detector.model = SimpleNamespace(predictor=SimpleNamespace(trackers=[SimpleNamespace(reset=lambda: calls.append(True))]))
    detector.reset()
    assert calls == [True] and detector.model.predictor is None

def test_empty_frame_rejected():
    detector = YOLODetector.__new__(YOLODetector)
    with pytest.raises(ValueError, match='empty'):
        detector.track(np.empty((0, 0, 3)))
