from collections import Counter, defaultdict
import math

class ObjectCounter:
    """Finite directed line, dead band, and once-per-ID-per-direction counting."""
    def __init__(self, line=None, tolerance=8):
        if line and math.dist(line[0], line[1]) == 0:
            raise ValueError('Counting line endpoints must differ.')
        self.line = line
        self.tolerance = tolerance
        self.unique = defaultdict(set)
        self.sides = {}
        self.counted = set()
        self.crossings = Counter()
        self.labels = {}

    def update(self, detections):
        for d in detections:
            if d.track_id is None or not d.confirmed:
                continue
            tid = d.track_id
            old_label = self.labels.get(tid)
            if old_label is not None and old_label != d.label:
                self.unique[old_label].discard(tid)
                for direction in ('A to B', 'B to A'):
                    if (tid, direction) in self.counted:
                        self.crossings[(old_label, direction)] -= 1
                        self.crossings[(d.label, direction)] += 1
            self.labels[tid] = d.label
            self.unique[d.label].add(tid)
            if self.line is None:
                continue
            a, b = self.line
            p = d.centroid
            dx, dy = b[0] - a[0], b[1] - a[1]
            distance = (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / math.hypot(dx, dy)
            if abs(distance) <= self.tolerance:
                continue
            side = 1 if distance > 0 else -1
            previous = self.sides.get(tid)
            if previous and previous[0] != side:
                old = previous[1]
                old_distance = dx * (old[1] - a[1]) - dy * (old[0] - a[0])
                new_distance = distance * math.hypot(dx, dy)
                t = old_distance / (old_distance - new_distance)
                intersection = (old[0] + t * (p[0] - old[0]), old[1] + t * (p[1] - old[1]))
                projection = ((intersection[0] - a[0]) * dx + (intersection[1] - a[1]) * dy) / (dx * dx + dy * dy)
                direction = 'A to B' if side > 0 else 'B to A'
                key = (tid, direction)
                if 0 <= projection <= 1 and key not in self.counted:
                    self.crossings[(d.label, direction)] += 1
                    self.counted.add(key)
            self.sides[tid] = (side, p)

    def counts(self):
        return [dict(object_class=label, unique_objects=len(ids),
                     a_to_b=self.crossings[(label, 'A to B')], b_to_a=self.crossings[(label, 'B to A')])
                for label, ids in sorted(self.unique.items()) if ids]
