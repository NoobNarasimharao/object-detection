from dataclasses import replace
import pytest
from src.detector import Detection
from src.tracker import TrackStabilizer
from src.video_processor import ProcessingConfig

def observation(label='car', score=.8, tid=1):
    classes = {'car':2, 'cell phone':67, 'dog':16, 'person':0}
    return Detection(tid, classes[label], label, score, (10,10,40,40))

def test_confirmation_and_low_confidence_recovery():
    tracker = TrackStabilizer(ProcessingConfig())
    assert not tracker.update([observation()],1)[0].confirmed
    assert not tracker.update([observation()],2)[0].confirmed
    assert tracker.update([observation()],3)[0].confirmed
    tracker.update([],4)
    recovered = tracker.update([observation(score=.15)],5)[0]
    assert recovered.confirmed and recovered.track_id == 1

def test_single_mistake_does_not_override_stable_label():
    tracker = TrackStabilizer(ProcessingConfig())
    for step in range(1,9): tracker.update([observation()],step)
    output = tracker.update([observation('cell phone', .99)],9)[0]
    assert output.label == 'car' and output.raw_label == 'cell phone'
    assert output.confidence == .99 and output.class_support < 1

def test_wrong_initial_label_expires_under_new_evidence():
    tracker = TrackStabilizer(ProcessingConfig())
    for step in range(1,9): tracker.update([observation('cell phone', .99)],step)
    for step in range(9,25): output = tracker.update([observation('car', .8)],step)[0]
    assert output.label == 'car'

def test_unreliable_and_duplicate_observations_do_not_confirm():
    tracker = TrackStabilizer(ProcessingConfig())
    for step in range(1,8):
        result = tracker.update([observation(score=.12)]*3,step)
        assert len(result)==1 and not result[0].confirmed
    other = TrackStabilizer(ProcessingConfig())
    assert not other.update([observation()]*5,1)[0].confirmed

def test_expired_state_and_reset():
    tracker = TrackStabilizer(ProcessingConfig(track_buffer=3))
    for step in range(1,4): tracker.update([observation()],step)
    assert not tracker.update([observation('dog')],8)[0].confirmed
    assert tracker.states[1]['label'][1]=='dog'
    tracker.reset()
    assert tracker.states == {}

def test_multiclass_independent_histories():
    tracker = TrackStabilizer(ProcessingConfig())
    for step in range(1,4):
        out = tracker.update([observation('dog',tid=1),replace(observation('person',tid=2),box=(50,50,90,90))],step)
    assert {d.label for d in out} == {'dog','person'}
    assert all(d.confirmed for d in out)

def test_persistent_weak_evidence_loses_confirmation():
    tracker = TrackStabilizer(ProcessingConfig())
    for step in range(1,4): tracker.update([observation()],step)
    for step in range(4,13): output = tracker.update([observation(score=.11)],step)[0]
    assert not output.confirmed

def test_duplicate_identity_is_auditable_and_does_not_confirm_twice():
    tracker = TrackStabilizer(ProcessingConfig())
    for step in range(1,5): tracker.update([observation(tid=1)],step)
    for step in range(5,9):
        out = tracker.update([observation(tid=1),observation('cell phone',.99,tid=2)],step)
    assert [d.track_id for d in out if d.confirmed] == [1]
    duplicate = next(d for d in out if d.track_id==2)
    assert duplicate.duplicate_of==1 and duplicate.raw_label=='cell phone'
    assert duplicate.track_id==2  # Not remapped to another identity.

def test_duplicate_suppression_can_be_disabled():
    tracker = TrackStabilizer(ProcessingConfig(duplicate_iou=0))
    for step in range(1,4): out=tracker.update([observation(tid=1),observation(tid=2)],step)
    assert sum(d.confirmed for d in out)==2

@pytest.mark.parametrize('backend', ['bytetrack', 'botsort'])
def test_real_tracker_retains_id_over_short_gap(backend):
    """Exercise real association using mock boxes, without a neural network/GPU."""
    import numpy as np
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.engine.results import Boxes
    args = SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                           track_buffer=30,match_thresh=.8,fuse_score=True,
                           with_reid=False,gmc_method='none',proximity_thresh=.5,appearance_thresh=.8)
    tracker = (BYTETracker if backend=='bytetrack' else BOTSORT)(args)
    ids = []
    image = np.zeros((100,100,3),np.uint8)
    for step in range(8):
        data = np.empty((0,6),np.float32) if step in (3,4) else np.array([[20+step,20,50+step,60,.9,0]],np.float32)
        tracks = tracker.update(Boxes(data,(100,100)),image)
        if len(tracks): ids.append(int(tracks[0,4]))
    assert len(ids)>=5 and len(set(ids))==1
