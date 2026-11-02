"""Readable, class-colored annotations; tracker identity stays in the label."""
from collections import Counter
import colorsys
import cv2
import numpy as np


def color(class_id):
    rgb = colorsys.hsv_to_rgb(((class_id or 0) * .61803398875) % 1, .7, 1)
    return tuple(round(v * 255) for v in reversed(rgb))


def text_box(canvas, text, x, y, foreground=(255, 255, 255), background=(22, 29, 40), scale=None):
    scale = scale or max(.45, min(.9, canvas.shape[1] / 1400))
    font = cv2.FONT_HERSHEY_SIMPLEX
    width, height = cv2.getTextSize(text, font, scale, 1)[0]
    # Long labels are shortened only visually; complete values stay in exports.
    while width > canvas.shape[1] - 8 and len(text) > 8:
        text = text[:-4] + '...'
        width, height = cv2.getTextSize(text, font, scale, 1)[0]
    x = max(2, min(int(x), canvas.shape[1] - width - 5))
    y = max(height + 3, min(int(y), canvas.shape[0] - 4))
    cv2.rectangle(canvas, (x-2, y-height-3), (x+width+2, y+3), background, -1)
    cv2.putText(canvas, text, (x, y), font, scale, foreground, 1, cv2.LINE_AA)


def annotate(frame, detections, history, counter, trails=True, centroids=True, show_summary=True, image_mode=False):
    canvas = frame.copy()
    for d in detections:
        c = color(d.class_id)
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
        label = f'{d.label} {d.confidence:.2f}'
        if not image_mode:
            label += f' ID:{d.track_id}' if d.confirmed else ' tentative'
            if d.raw_label and d.raw_label != d.label:
                label = f'{d.label} ID:{d.track_id} (raw {d.raw_label} {d.confidence:.2f})'
        text_box(canvas, label, x1, y1-7 if y1 > 22 else y1+18, foreground=c)
        if centroids:
            cv2.circle(canvas, tuple(map(int, d.centroid)), 3, c, -1)
        points = history.trails.get(d.track_id, [])
        if trails and len(points) > 1:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, c, 2)
    if counter.line:
        a, b = [tuple(map(int, p)) for p in counter.line]
        cv2.arrowedLine(canvas, a, b, (0, 230, 255), 2)
    if show_summary:
        visible = Counter(d.label for d in detections if image_mode or d.confirmed)
        total = sum(map(len, counter.unique.values()))
        heading = f'Detections: {len(detections)}' if image_mode else f'Confirmed now: {sum(visible.values())} | Unique IDs: {total}'
        if counter.line:
            heading += f' | Crossings: {sum(counter.crossings.values())}'
        text_box(canvas, heading, 6, canvas.shape[0]-max(27, round(canvas.shape[1]/1400*30)))
        text_box(canvas, 'Visible: ' + (', '.join(f'{label}: {n}' for label,n in sorted(visible.items())) or 'none'), 6, canvas.shape[0]-7)
    return canvas
