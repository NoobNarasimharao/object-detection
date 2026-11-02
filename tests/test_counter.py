from src.counter import ObjectCounter
from src.detector import Detection

def det(x, y, tid=1, label='person'):
    return Detection(tid, 0, label, .9, (x-1, y-1, x+1, y+1))

def test_unique_and_untracked():
    c = ObjectCounter()
    c.update([det(2, 2), det(2, 2), det(3, 3, None)])
    assert c.counts()[0]['unique_objects'] == 1

def test_directions_deadband_and_repeat():
    c = ObjectCounter(((0, 10), (20, 10)), tolerance=2)
    for y in [0, 9, 11, 14, 9, 0, 14, 0]:
        c.update([det(10, y)])
    assert c.crossings[('person', 'A to B')] == 1
    assert c.crossings[('person', 'B to A')] == 1

def test_outside_line_segment():
    c = ObjectCounter(((0, 10), (20, 10)), tolerance=1)
    c.update([det(30, 0)])
    c.update([det(30, 20)])
    assert not c.crossings

def test_diagonal_and_classes():
    c = ObjectCounter(((0, 0), (20, 20)), tolerance=1)
    c.update([det(15, 5), det(15, 5, 2, 'car')])
    c.update([det(5, 15), det(5, 15, 2, 'car')])
    assert sum(c.crossings.values()) == 2

def test_reclassification_does_not_double_count_identity_or_crossings():
    c = ObjectCounter(((0,10),(20,10)), tolerance=1)
    c.update([det(10,0,label='cell phone')])
    c.update([det(10,20,label='cell phone')])
    c.update([det(10,20,label='car')])
    assert sum(map(len,c.unique.values())) == 1
    assert c.counts() == [dict(object_class='car', unique_objects=1, a_to_b=1, b_to_a=0)]
    assert sum(c.crossings.values()) == 1

def test_tentative_tracks_do_not_count():
    from dataclasses import replace
    c = ObjectCounter()
    c.update([replace(det(10,10), confirmed=False)])
    assert c.counts() == []
