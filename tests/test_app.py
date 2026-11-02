from streamlit.testing.v1 import AppTest
from pathlib import Path
import pytest

def test_dashboard_launches_without_model_download():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == 'MotionLab'
    assert app.button[0].disabled
    assert next(s for s in app.selectbox if s.label == 'Scene classes').value == 'All objects'
    assert next(s for s in app.toggle if s.label == 'Count line crossings').value is False

@pytest.mark.parametrize('image_mode', [True, False])
def test_dashboard_displays_new_results(tmp_path, image_mode):
    import cv2
    import numpy as np
    from src.video_processor import ProcessingConfig, VideoProcessor
    from tests.test_video_processor import FakeDetector
    frame = np.zeros((48,64,3), np.uint8)
    if image_mode:
        source = tmp_path / 'input.png'
        cv2.imwrite(str(source),frame)
    else:
        source = tmp_path / 'input.avi'
        writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
        assert writer.isOpened()
        for _ in range(4): writer.write(frame)
        writer.release()
    result = VideoProcessor(ProcessingConfig(),FakeDetector()).process(source,tmp_path/'out')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py',default_timeout=60)
    app.session_state['result'] = result
    app.run()
    assert not app.exception

def test_scene_preset_can_return_to_all_objects():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py',default_timeout=60).run()
    preset = next(s for s in app.selectbox if s.label == 'Scene classes')
    preset.select('Road vehicles').run()
    assert not app.exception
    next(s for s in app.selectbox if s.label == 'Scene classes').select('All objects').run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == 'Scene classes').value == 'All objects'
