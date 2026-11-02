from dataclasses import dataclass, asdict, replace
from pathlib import Path
from threading import Event, Lock
import math
import subprocess
import time
import uuid
import cv2
from .detector import YOLODetector
from .tracker import TrackHistory, TrackStabilizer
from .counter import ObjectCounter
from .visualizer import annotate
from .analytics import export_results
from .evaluator import evaluate
from .database import Database

@dataclass
class ProcessingConfig:
    model: str = 'yolo26s.pt'
    tracker: str = 'bytetrack.yaml'
    confidence: float = .1
    display_confidence: float = .25
    iou: float = .5
    imgsz: int = 640
    device: str = 'auto'
    classes: list[int] | None = None
    trail_length: int = 40
    show_trails: bool = True
    centroids: bool = True
    counting: bool = False
    line: tuple = ((.1, .5), (.9, .5))
    tolerance: float = 8
    stride: int = 1
    max_width: int = 0
    max_frames: int = 0
    min_track_hits: int = 3
    confirmation_window: int = 8
    class_window: int = 15
    class_switch_hits: int = 3
    stabilize_classes: bool = True
    track_buffer: int = 30
    match_threshold: float = .8
    show_tentative: bool = False
    show_summary: bool = True
    duplicate_iou: float = .95

    def __post_init__(self):
        if not 0 < self.confidence <= 1 or not 0 <= self.iou <= 1:
            raise ValueError('Confidence and IoU must be in their valid probability ranges.')
        if not 0 <= self.display_confidence <= 1:
            raise ValueError('Display confidence must be between zero and one.')
        if self.stride < 1 or self.imgsz < 32 or self.trail_length < 1 or (self.max_width != 0 and self.max_width < 32) or self.max_frames < 0:
            raise ValueError('Invalid frame, resolution or trail configuration.')
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('Line tolerance must be finite and nonnegative.')
        if self.classes and any(class_id < 0 or class_id > 79 for class_id in self.classes):
            raise ValueError('Pretrained COCO class IDs must be between 0 and 79.')
        if self.tracker not in ('bytetrack.yaml', 'botsort.yaml'):
            raise ValueError('Select bytetrack.yaml or botsort.yaml.')
        if len(self.line) != 2 or any(len(p) != 2 for p in self.line) or any(not 0 <= x <= 1 for p in self.line for x in p):
            raise ValueError('Line coordinates must be normalized to [0,1].')
        if self.counting and self.line[0] == self.line[1]:
            raise ValueError('Line endpoints must differ.')
        if not 1 <= self.min_track_hits <= self.confirmation_window <= self.class_window:
            raise ValueError('Require 1 <= confirmation hits <= confirmation window <= class window.')
        if not 1 <= self.class_switch_hits <= self.class_window or self.track_buffer < 1 or not 0 <= self.match_threshold <= 1:
            raise ValueError('Invalid class voting or tracker association settings.')
        if not 0 <= self.duplicate_iou <= 1:
            raise ValueError('Duplicate IoU must be between zero and one; zero disables suppression.')


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def output_frame(frame, max_width=0, video=True):
    """Resize only the rendered output. Inference always receives original pixels."""
    h, w = frame.shape[:2]
    scale = min(1., max_width / w) if max_width else 1.
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    rendered = frame if size == (w, h) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    # Pad an odd edge instead of stretching to satisfy H.264's even dimensions.
    if video:
        rendered = cv2.copyMakeBorder(rendered, 0, size[1] % 2, 0, size[0] % 2, cv2.BORDER_REPLICATE)
    return rendered, (size[0] / w, size[1] / h)

def scale_detections(detections, scale):
    sx, sy = scale
    return [replace(d, box=(d.box[0]*sx, d.box[1]*sy, d.box[2]*sx, d.box[3]*sy)) for d in detections]

def record(d, index, timestamp):
    return dict(track_id=d.track_id, object_class=d.label, confidence=d.confidence,
                **dict(zip(['x1', 'y1', 'x2', 'y2'], d.box)), centroid_x=d.centroid[0], centroid_y=d.centroid[1],
                frame_number=index, timestamp=timestamp, raw_class=d.raw_label or d.label,
                raw_class_id=d.raw_class_id if d.raw_class_id is not None else d.class_id,
                class_support=d.class_support, confirmed=d.confirmed, duplicate_of=d.duplicate_of)

class Job:
    """Thread-safe progress shared with Streamlit; worker never calls st APIs."""
    def __init__(self):
        self.cancel = Event()
        self.lock = Lock()
        self.state = dict(status='Starting', processed=0, total=0, preview=None, result=None, error=None, done=False)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return self.state.copy()

def transcode(raw, final):
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw), '-an',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(final)],
                       check=True, capture_output=True, timeout=600)
        raw.unlink()
        return final, None
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return raw, f'H.264 conversion unavailable ({type(exc).__name__}); download the MPEG-4 video and play in VLC, or convert with FFmpeg.'

class VideoProcessor:
    def __init__(self, config, detector=None):
        self.config = config
        self.detector = detector

    def process(self, source, output='data/output', job=None):
        job = job or Job()
        c = self.config
        is_camera = isinstance(source, int)
        if not is_camera and Path(source).suffix.lower() in IMAGE_SUFFIXES:
            return self.process_image(source, output, job)
        if not is_camera:
            source = Path(source)
            if not source.is_file():
                raise ValueError(f'Video does not exist: {source}')
            if source.suffix.lower() not in ('.mp4', '.avi', '.mov', '.mkv'):
                raise ValueError('Supported inputs: MP4, AVI, MOV and MKV (decoder permitting).')
        cap = cv2.VideoCapture(source if is_camera else str(source))
        writer = None
        started = time.perf_counter()
        records, frames, confidence, warnings, observations = [], [], [], [], []
        history = TrackHistory(c.trail_length)
        stabilizer = TrackStabilizer(c)
        counter = None
        run_id = uuid.uuid4().hex
        folder = Path(output) / run_id
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open video or local webcam. Check file decoding, camera index and permissions.')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
                warnings.append('Source FPS unavailable; output uses 30 FPS before frame skipping.')
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
            folder.mkdir(parents=True, exist_ok=False)
            detector = self.detector or YOLODetector(c)
            detector.reset()
            if hasattr(detector, 'configure_tracking'):
                detector.configure_tracking(folder)
            raw = folder / 'annotated_mpeg4.mp4'
            index = 0
            camera_start = time.perf_counter()
            while not job.cancel.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_camera:
                        warnings.append('Camera stream ended or failed to supply a frame.')
                    elif total > 0 and index < total - 1:
                        warnings.append('Decoding ended before the reported frame count; output may be partial.')
                    break
                index += 1
                if frame is None or frame.size == 0:
                    raise ValueError('Decoder returned an empty frame.')
                if (index - 1) % c.stride:
                    continue
                tick = time.perf_counter()
                h, w = frame.shape[:2]
                original = frame
                frame, scale = output_frame(original, c.max_width)
                size = (frame.shape[1], frame.shape[0])
                if not frames and min(h, w) < 360:
                    warnings.append('Low-resolution input: higher inference size cannot restore missing image detail.')
                if writer is None:
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps / c.stride, size)
                    if not writer.isOpened():
                        raise RuntimeError('Unable to create MP4 video. Check output permissions and OpenCV codecs.')
                    line = tuple((x * (size[0] - 1), y * (size[1] - 1)) for x, y in c.line) if c.counting else None
                    counter = ObjectCounter(line, c.tolerance)
                inference_start = time.perf_counter()
                raw_detections = detector.track(original)
                inference_ms = (time.perf_counter() - inference_start) * 1000
                assessed = stabilizer.update(raw_detections, len(frames) + 1)
                assessed += [replace(d, confirmed=False) for d in raw_detections if d.track_id is None]
                assessed = scale_detections(assessed, scale)
                detections = [d for d in assessed if d.confirmed or (c.show_tentative and d.confidence >= c.display_confidence)]
                timestamp = time.perf_counter() - camera_start if is_camera else (index - 1) / fps
                history.update(detections, index, timestamp)
                counter.update(detections)
                canvas = annotate(frame, detections, history, counter, c.show_trails, c.centroids, c.show_summary)
                writer.write(canvas)
                observations.extend(record(d, index, timestamp) for d in assessed)
                for d in detections:
                    confidence.append(d.confidence)
                    records.append(record(d, index, timestamp))
                frames.append(dict(frame_number=index, timestamp=timestamp, visible_objects=len(detections),
                    tracked_objects=sum(d.track_id is not None and d.confirmed for d in detections), inference_ms=inference_ms,
                    processing_fps=1 / max(time.perf_counter() - tick, 1e-9)))
                job.update(status='Processing', processed=len(frames), total=math.ceil(total / c.stride),
                           preview=cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                if c.max_frames and len(frames) >= c.max_frames:
                    break
            if not frames:
                raise ValueError('No frames processed. The video is empty, corrupted, or processing was cancelled before the first frame.')
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        processing_elapsed = time.perf_counter() - started
        job.update(status='Exporting')
        video, warning = transcode(raw, folder / 'annotated.mp4')
        if warning:
            warnings.append(warning)
        summary = evaluate(frames, confidence, processing_elapsed)
        summary.update(run_id=run_id, source=str(source), configuration=asdict(c),
                       unique_objects=sum(map(len, counter.unique.values())),
                       currently_visible_objects=frames[-1]['visible_objects'],
                       total_crossings=sum(counter.crossings.values()), counts=counter.counts(),
                       cancelled=job.cancel.is_set(), warnings=warnings, output_video=str(video),
                       source_fps=fps, output_fps=fps / c.stride, output_size=size, media_type='video',
                       source_size=[w, h], total_observations=len(observations))
        export_results(folder, records, frames, counter.counts(), history.summaries(), summary, observations)
        Database(Path(output) / 'history.db').save(run_id, source, folder, summary)
        return dict(folder=str(folder), video=str(video), summary=summary)

    def process_image(self, source, output='data/output', job=None):
        """Single-image detection uses no synthetic track IDs or crossing counts."""
        job = job or Job()
        c = self.config
        if job.cancel.is_set():
            raise ValueError('Image processing was cancelled.')
        frame = cv2.imread(str(source))
        if frame is None or frame.size == 0:
            raise ValueError('Cannot open image. Check file existence, permissions and decoder support.')
        started = time.perf_counter()
        run_id = uuid.uuid4().hex
        folder = Path(output) / run_id
        folder.mkdir(parents=True, exist_ok=False)
        detector = self.detector or YOLODetector(c)
        detector.reset()
        tick = time.perf_counter()
        raw = [replace(d, track_id=None, confirmed=False) for d in detector.predict(frame)]
        inference_ms = (time.perf_counter() - tick) * 1000
        canvas, scale = output_frame(frame, c.max_width, video=False)
        raw = scale_detections(raw, scale)
        detections = [d for d in raw if d.confidence >= c.display_confidence]
        from collections import Counter
        counts = Counter(d.label for d in detections)
        count_rows = [dict(object_class=label, unique_objects=0, a_to_b=0, b_to_a=0, image_detections=count)
                      for label, count in sorted(counts.items())]
        canvas = annotate(canvas, detections, TrackHistory(), ObjectCounter(), False, c.centroids, c.show_summary, image_mode=True)
        path = folder / 'annotated.jpg'
        if not cv2.imwrite(str(path), canvas):
            raise OSError('Could not save annotated image.')
        elapsed = time.perf_counter() - started
        frames = [dict(frame_number=1, timestamp=0., visible_objects=len(detections), tracked_objects=0,
                       inference_ms=inference_ms, processing_fps=1 / max(elapsed, 1e-9))]
        summary = evaluate(frames, [d.confidence for d in detections], elapsed)
        summary.update(run_id=run_id, source=str(source), configuration=asdict(c), media_type='image',
                       unique_objects=0, currently_visible_objects=len(detections), total_crossings=0,
                       counts=count_rows, image_counts=dict(counts), cancelled=False, warnings=[], output_image=str(path),
                       output_size=[canvas.shape[1], canvas.shape[0]], source_size=[frame.shape[1], frame.shape[0]])
        export_results(folder, [record(d, 1, 0.) for d in detections], frames, count_rows, [], summary,
                       [record(d, 1, 0.) for d in raw])
        Database(Path(output) / 'history.db').save(run_id, source, folder, summary)
        job.update(processed=1, total=1, preview=cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return dict(folder=str(folder), video=None, image=str(path), summary=summary)

def run_job(source, output, config, job):
    try:
        result = VideoProcessor(config).process(source, output, job)
        job.update(result=result, status='Cancelled (partial output saved)' if job.cancel.is_set() else 'Complete')
    except Exception as exc:
        job.update(error=f'{type(exc).__name__}: {exc}', status='Failed')
    finally:
        job.update(done=True)
