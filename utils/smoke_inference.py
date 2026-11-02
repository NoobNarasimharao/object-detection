"""Optional online smoke check: real YOLO inference on a repeated reference image.

Run with `.venv/Scripts/python.exe -m utils.smoke_inference`.
This verifies integration, not accuracy on moving video.
"""
from pathlib import Path
import json
import urllib.request
import cv2
import pandas as pd
from src.video_processor import ProcessingConfig, VideoProcessor
from .helpers import ROOT

def main():
    folder = ROOT / 'data/input'
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / 'bus.jpg'
    if not image_path.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg', image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError('Reference image did not decode.')
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (width // 2 * 2, height // 2 * 2))
    source = folder / 'smoke_reference.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5, (frame.shape[1], frame.shape[0]))
    try:
        if not writer.isOpened():
            raise RuntimeError('Unable to write smoke input.')
        for _ in range(6):
            writer.write(frame)
    finally:
        writer.release()
    result = VideoProcessor(ProcessingConfig(imgsz=320, device='cpu')).process(source, ROOT / 'data/output')
    records = pd.read_csv(Path(result['folder']) / 'tracking.csv')
    assert len(records) > 0, 'Expected detections in the bus reference image.'
    assert records.track_id.notna().any(), 'ByteTrack produced no IDs.'
    assert records.groupby('track_id').frame_number.nunique().max() > 1, 'No ID persisted across frames.'
    capture = cv2.VideoCapture(result['video'])
    try:
        assert capture.isOpened() and capture.read()[0]
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    finally:
        capture.release()
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
