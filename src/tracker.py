from collections import deque, Counter
from dataclasses import replace


class TrackStabilizer:
    """Confirm tracker-owned identities and smooth classes without inventing IDs.

    All windows and gaps use processed-frame indices, not source frame indices.
    Weighted votes expire, so persistent new evidence can replace an old label.
    No predicted or missing boxes are synthesized during occlusion.
    """
    def __init__(self, config):
        self.config = config
        self.states = {}

    def reset(self):
        self.states.clear()

    def update(self, detections, step):
        c = self.config
        self.states = {tid: state for tid, state in self.states.items()
                       if step - state['last'] <= c.track_buffer}
        # A repeated ID in one frame must not confirm itself multiple times.
        best = {}
        for d in detections:
            if d.track_id is not None and (d.track_id not in best or d.confidence > best[d.track_id].confidence):
                best[d.track_id] = d
        output = []
        for tid, d in best.items():
            state = self.states.setdefault(tid, dict(first=step, last=step, history=deque(maxlen=c.class_window),
                label=(d.class_id, d.label), confirmed=False))
            history = state['history']
            if not history or history[-1][0] != step:
                history.append((step, d.class_id, d.label, d.confidence))
            state['last'] = step
            # Old evidence also expires across missing frames.
            while history and step - history[0][0] >= c.class_window:
                history.popleft()
            votes = Counter()
            hits = Counter()
            for when, cls, label, confidence in history:
                votes[(cls, label)] += confidence * (.9 ** (step - when))
                hits[(cls, label)] += 1
            winner = max(votes, key=votes.get)
            current = state['label']
            if c.stabilize_classes:
                if winner != current and hits[winner] >= c.class_switch_hits and votes[winner] > votes[current] * 1.25:
                    state['label'] = winner
            else:
                state['label'] = (d.class_id, d.label)
            recent = [obs for obs in history if step - obs[0] < c.confirmation_window]
            strong = sum(obs[3] >= c.display_confidence for obs in recent)
            if strong == 0:
                state['confirmed'] = False
            if (len(recent) >= c.min_track_hits and strong >= min(2, c.min_track_hits)
                    and votes[state['label']] / max(sum(votes.values()), 1e-9) >= .6):
                state['confirmed'] = True
            cls, label = state['label']
            support = votes[(cls, label)] / max(sum(votes.values()), 1e-9)
            output.append(replace(d, class_id=cls, label=label, raw_class_id=d.class_id, raw_label=d.label,
                                  class_support=support, confirmed=state['confirmed']))
        if not c.duplicate_iou:
            return output
        # Different IDs can describe the same near-identical box, particularly
        # when class-aware NMS retains conflicting labels. Never invent or merge IDs.
        kept, assessed = [], []
        for d in sorted(output, key=lambda d: (not d.confirmed, self.states[d.track_id]['first'], -d.confidence)):
            duplicate = next((other for other in kept if box_iou(d.box, other.box) >= c.duplicate_iou), None)
            if duplicate is not None:
                assessed.append(replace(d, confirmed=False, duplicate_of=duplicate.track_id))
            else:
                kept.append(d)
                assessed.append(d)
        return assessed


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.

class TrackHistory:
    def __init__(self, length=40):
        self.length = length
        self.reset()

    def reset(self):
        self.trails = {}
        self.tracks = {}

    def update(self, detections, frame, timestamp):
        for d in detections:
            if d.track_id is None or not d.confirmed:
                continue
            tid = d.track_id
            self.trails.setdefault(tid, deque(maxlen=self.length)).append(d.centroid)
            info = self.tracks.setdefault(tid, dict(track_id=tid, object_class=d.label,
                entry_frame=frame, entry_time=timestamp, last_seen_frame=frame,
                last_seen_time=timestamp, observations=0))
            info.update(object_class=d.label, last_seen_frame=frame, last_seen_time=timestamp, observations=info['observations'] + 1)

    def summaries(self):
        return [dict(info, duration_seconds=info['last_seen_time'] - info['entry_time']) for info in self.tracks.values()]
