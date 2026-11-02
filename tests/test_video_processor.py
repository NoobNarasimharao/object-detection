import json
import cv2
import numpy as np
import pandas as pd
import pytest
from src.video_processor import ProcessingConfig, VideoProcessor, Job
from src.database import Database
from tests.test_counter import det

class FakeDetector:
    def __init__(self, empty=False): self.resets = 0; self.empty = empty
    def reset(self): self.resets += 1
    def track(self, frame): return [] if self.empty else [det(20, 20)]
    def predict(self, frame): return self.track(frame)

@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(6): writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path

@pytest.mark.parametrize('empty', [False, True])
def test_output_exports_reset(video, tmp_path, empty):
    detector = FakeDetector(empty)
    processor = VideoProcessor(ProcessingConfig(stride=2), detector)
    result = processor.process(video, tmp_path / 'out')
    assert detector.resets == 1
    assert result['summary']['frames_processed'] == 3
    assert result['summary']['output_fps'] == 5
    from pathlib import Path
    folder = Path(result['folder'])
    records = pd.read_csv(folder / 'tracking.csv')
    assert len(records) == (0 if empty else 1)  # Three observations before confirmation.
    observations = pd.read_csv(folder / 'observations.csv')
    assert len(observations) == (0 if empty else 3)
    assert json.loads((folder / 'analytics.json').read_text())['unique_objects'] == (0 if empty else 1)
    capture = cv2.VideoCapture(result['video'])
    try:
        assert capture.isOpened() and capture.read()[0]
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()
    assert len(Database(tmp_path / 'out/history.db').recent()) == 1
    processor.process(video, tmp_path / 'out')
    assert detector.resets == 2

def test_invalid_input(tmp_path):
    p = VideoProcessor(ProcessingConfig(), FakeDetector())
    with pytest.raises(ValueError): p.process(tmp_path / 'missing.mp4', tmp_path)
    corrupt = tmp_path / 'bad.mp4'
    corrupt.write_bytes(b'not a video')
    with pytest.raises(ValueError): p.process(corrupt, tmp_path)

def test_cancel_before_frames(video, tmp_path):
    job = Job()
    job.cancel.set()
    with pytest.raises(ValueError, match='No frames'):
        VideoProcessor(ProcessingConfig(), FakeDetector()).process(video, tmp_path / 'out', job)

def test_ffmpeg_fallback(tmp_path, monkeypatch):
    from src.video_processor import transcode
    import imageio_ffmpeg
    def unavailable(): raise RuntimeError('Unavailable')
    monkeypatch.setattr(imageio_ffmpeg, 'get_ffmpeg_exe', unavailable)
    raw = tmp_path / 'raw.mp4'
    raw.write_bytes(b'video')
    path, warning = transcode(raw, tmp_path / 'final.mp4')
    assert path == raw and path.exists() and 'H.264' in warning

@pytest.mark.parametrize('kwargs', [{'stride':0}, {'confidence':2}, {'counting':True, 'line':((.5,.5),(.5,.5))}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): ProcessingConfig(**kwargs)

def test_image_output_has_no_fake_ids(tmp_path):
    from pathlib import Path
    image = tmp_path / 'scene.png'
    cv2.imwrite(str(image), np.zeros((41, 81, 3), dtype=np.uint8))
    result = VideoProcessor(ProcessingConfig(), FakeDetector()).process(image, tmp_path / 'out')
    assert result['video'] is None
    assert cv2.imread(result['image']).shape[:2] == (41, 81)
    assert result['summary']['image_counts'] == {'person': 1}
    assert result['summary']['unique_objects'] == 0
    records = pd.read_csv(Path(result['folder']) / 'tracking.csv')
    assert records.track_id.isna().all()
    assert not records.confirmed.any()
    assert pd.read_csv(Path(result['folder']) / 'counts.csv').image_detections.sum() == 1

def test_inference_keeps_original_pixels(video, tmp_path):
    shapes = []
    class Recorder(FakeDetector):
        def track(self, frame):
            shapes.append(frame.shape[:2])
            return super().track(frame)
    result = VideoProcessor(ProcessingConfig(max_width=32), Recorder()).process(video, tmp_path / 'out')
    assert shapes == [(48, 64)] * 6
    assert result['summary']['output_size'] == (32, 24)
    from pathlib import Path
    records = pd.read_csv(Path(result['folder']) / 'tracking.csv')
    assert (records.centroid_x == 10).all()

def test_odd_video_dimensions_are_padded_not_stretched():
    from src.video_processor import output_frame
    frame = np.arange(5*7*3, dtype=np.uint8).reshape(5,7,3)
    output, scale = output_frame(frame)
    assert output.shape == (6,8,3) and scale == (1,1)
    assert np.array_equal(output[:5,:7], frame)
