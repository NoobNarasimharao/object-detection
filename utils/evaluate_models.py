"""Reproducible, unlabelled model comparison; never infer accuracy from scores.

Example: python -m utils.evaluate_models --images data/input/bus.jpg --video clip.mp4
Use clean original inputs. Mark annotated sources explicitly in the analysis report.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from src.detector import extract_detections
from src.video_processor import ProcessingConfig, VideoProcessor
from src.visualizer import annotate
from src.tracker import TrackHistory
from src.counter import ObjectCounter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='*', default=[])
    parser.add_argument('--video')
    parser.add_argument('--output', default='data/output/model_evaluation')
    parser.add_argument('--sequence-frames', type=int, default=160)
    args = parser.parse_args()
    from ultralytics import YOLO
    import ultralytics
    import torch
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    samples = []
    for path in map(Path, args.images):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot read {path}')
        samples.append((path.stem, str(path), frame))
    if args.video:
        capture = cv2.VideoCapture(args.video)
        try:
            if not capture.isOpened():
                raise ValueError('Cannot read evaluation video')
            for index in [30, 60, 120, 200, 380]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    samples.append((f'video_{index:04}', f'{args.video}#frame={index}', frame))
        finally:
            capture.release()
    if not samples:
        parser.error('Provide at least one image or a video')
    reports = []
    rows = []
    for model_name, imgsz in [('yolo26n.pt', 640), ('yolo26s.pt', 640), ('yolo26s.pt', 960)]:
        model = YOLO(model_name)
        name = f'{Path(model_name).stem}_{imgsz}'
        options = dict(conf=.1, iou=.5, imgsz=imgsz, device='cpu', classes=None, nms=None, verbose=False)
        model.predict(samples[0][2], **options)  # Warm up; exclude from timings.
        for sample, source, frame in samples:
            times = []
            for _ in range(3):
                start = time.perf_counter()
                result = model.predict(frame, **options)[0]
                times.append((time.perf_counter()-start)*1000)
            detections = extract_detections(result, frame.shape[1], frame.shape[0])
            shown = [d for d in detections if d.confidence >= .25]
            canvas = annotate(frame, shown, TrackHistory(), ObjectCounter(), False, False, True, image_mode=True)
            # Scale only saved review examples to keep them convenient to inspect.
            if canvas.shape[1] > 1100:
                canvas = cv2.resize(canvas, (1100, round(canvas.shape[0]*1100/canvas.shape[1])))
            cv2.imwrite(str(folder / f'{sample}_{name}.jpg'), canvas)
            report = dict(variant=name, source=source, sample=sample, source_shape=list(frame.shape),
                          mean_call_ms=float(np.mean(times)), detections=[asdict(d) for d in detections],
                          reported_classes=dict(Counter(d.label for d in shown)))
            reports.append(report)
            rows.append(dict(variant=name, sample=sample, mean_call_ms=report['mean_call_ms'],
                             reported_detections=len(shown), classes=json.dumps(report['reported_classes'])))
        print(f'Completed representative frames: {name}', flush=True)
    metadata = dict(ultralytics=ultralytics.__version__, torch=torch.__version__, cuda=torch.cuda.is_available(),
                    device='cpu', confidence=.1, comparison_reporting_threshold=.25, iou=.5,
                    classes='all', timing='Three warm calls per frame, includes pre/postprocess, excludes model load',
                    accuracy_metrics=None, reason='No ground-truth box or identity annotations supplied')
    (folder/'predictions.json').write_text(json.dumps(dict(metadata=metadata, samples=reports),indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(folder/'comparison.csv',index=False)
    if args.video and args.sequence_frames:
        sequence_results = []
        for model_name in ['yolo26n.pt', 'yolo26s.pt']:
            result = VideoProcessor(ProcessingConfig(model=model_name, max_frames=args.sequence_frames,
                    device='cpu', imgsz=640)).process(args.video, folder/'sequences')
            observations = pd.read_csv(Path(result['folder'])/'observations.csv')
            labeled = observations.dropna(subset=['track_id'])
            raw_changes = sum(int((group.raw_class != group.raw_class.shift()).sum())-1
                              for _,group in labeled.groupby('track_id'))
            stable_changes = sum(int((group.object_class != group.object_class.shift()).sum())-1
                                 for _,group in labeled.groupby('track_id'))
            sequence_results.append(dict(model=model_name, result=result, raw_class_changes=raw_changes,
                stabilized_class_changes=stable_changes, observed_ids=int(labeled.track_id.nunique()),
                suppressed_duplicate_observations=int(observations.duplicate_of.notna().sum()),
                note='Label transitions and observed IDs are diagnostics, not ground-truth ID switches or accuracy.'))
            print(f'Completed sequence: {model_name}',flush=True)
        (folder/'sequence_comparison.json').write_text(json.dumps(sequence_results,indent=2),encoding='utf-8')
    print(f'Evaluation artifacts: {folder.resolve()}')

if __name__ == '__main__':
    main()
