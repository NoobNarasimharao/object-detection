from pathlib import Path
import tempfile
from threading import Thread
import pandas as pd
import plotly.express as px
import streamlit as st
from src.database import Database
from src.video_processor import ProcessingConfig, Job, run_job, IMAGE_SUFFIXES
from utils.helpers import ROOT, defaults

st.set_page_config(page_title='MotionLab • YOLO + ByteTrack', page_icon='🎯', layout='wide')
st.title('MotionLab')
st.caption('REAL-TIME MULTI-OBJECT DETECTION AND TRACKING • COMPUTER VISION LAB')
d = defaults()
with st.sidebar:
    st.header('Processing studio')
    models = ['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt']
    model = st.selectbox('YOLO26 model', models, index=models.index(d['model']))
    st.caption('Nano: fastest CPU option. Small: larger general-purpose model to compare. Medium/large/x: more compute and memory; no guaranteed improvement on every scene.')
    source_type = st.radio('Video source', ['Upload video or image', 'Local server webcam'])
    confidence = st.slider('Tracker candidate confidence', .01, 1.0, float(d['confidence']), .01)
    st.caption('Low confidence candidates help ByteTrack recover occluded tracks. Raising this can reduce recovery.')
    display_confidence = st.slider('Confirmation evidence / image confidence', .0, 1.0, float(d['display_confidence']), .05)
    st.caption('New video tracks need repeated evidence. Confirmed tracks keep low-score recovery observations. For still images this is the detection cutoff. High confidence can still be wrong.')
    iou = st.slider('NMS IoU threshold', .0, 1.0, float(d['iou']), .05)
    from ultralytics.utils import YAML
    from ultralytics.utils import ROOT as ULTRALYTICS_ROOT
    @st.cache_data
    def class_names():
        return YAML.load(ULTRALYTICS_ROOT / 'cfg/datasets/coco.yaml')['names']
    names = class_names()
    preset = st.selectbox('Scene classes', ['All objects', 'Road vehicles', 'Custom classes'])
    if preset == 'Road vehicles':
        selected = [1, 2, 3, 5, 7]
        st.caption('Bicycle, car, motorcycle, bus and truck only. Other predictions are excluded, not relabeled as cars.')
    elif preset == 'Custom classes':
        selected = st.multiselect('Classes (empty = all 80)', list(names), format_func=lambda n: names[n])
    else:
        selected = []
    trails = st.toggle('Show trajectories', True)
    trail_length = st.slider('Trajectory length', 2, 150, 40)
    centroids = st.toggle('Show centroids', True)
    counting = st.toggle('Count line crossings', bool(d['counting']))
    show_summary = st.toggle('Show count overlay', True)
    device = st.selectbox('Device', ['auto', 'cpu', '0'])
    imgsz = st.select_slider('Inference resolution', [320, 480, 640, 960, 1280], value=640)
    max_width = st.selectbox('Maximum output width', [0, 640, 960, 1280, 1920], format_func=lambda x: 'Original size' if x == 0 else str(x))
    st.caption('Output resizing does not reduce the image supplied to inference. Low-resolution input cannot gain real detail by upscaling.')
    stride = st.slider('Process every Nth frame', 1, 10, 1)
    with st.expander('Counting line & advanced'):
        st.caption('Coordinates are fractions of image size. For a left-to-right line, A is above and B below. Each ID counts at most once per direction.')
        x1 = st.slider('Start X', 0., 1., .1)
        y1 = st.slider('Start Y', 0., 1., .5)
        x2 = st.slider('End X', 0., 1., .9)
        y2 = st.slider('End Y', 0., 1., .5)
        tolerance = st.slider('Dead band (output pixels)', 0, 40, 8)
        tracker = st.selectbox('Tracker', ['bytetrack.yaml', 'botsort.yaml'])
        st.caption('ByteTrack for a simple baseline; BoT-SORT adds camera-motion compensation. Neither guarantees identity through occlusion.')
        min_track_hits = st.slider('Observations to confirm a track', 1, 8, d['min_track_hits'])
        stabilize_classes = st.toggle('Stabilize classes from track history', True)
        class_window = st.slider('Class history (processed frames)', 8, 60, d['class_window'])
        track_buffer = st.slider('Lost-track buffer (processed frames)', 1, 120, d['track_buffer'])
        match_threshold = st.slider('Association cost limit', .1, 1., .8, .05)
        duplicate_iou = st.slider('Near-identical box suppression IoU (0 = off)', .0, 1., .95, .05)
        show_tentative = st.toggle('Show tentative detections (excluded from counts)', False)
        max_frames = st.number_input('Frame limit (0 = whole video)', 0, 100000, 0)

home, detection, analytics, tracking, info = st.tabs(['Home', 'Detection and Tracking', 'Analytics', 'Tracking Data', 'Project Information'])
with home:
    st.subheader('Turn video into measurable motion')
    st.write('Detect objects, follow their paths, and measure traffic through a virtual line. Each independent run owns its model and tracker, keeping IDs isolated between sessions.')
    a, b, c = st.columns(3)
    a.metric('Detector', 'YOLO26')
    b.metric('Primary tracker', 'ByteTrack')
    c.metric('Exports', 'MP4 · CSV · JSON')
    st.code('Video → Frames → YOLO → ByteTrack → Trajectories & Counts → Analytics & Exports', language=None)
    st.info('All 80 COCO classes are enabled by default. Small at 640 pixels is the balanced default; choose nano for faster CPU inference. Higher resolution is optional and can also add false positives. Upload a video or image.')

def launch(source, config, temporary=False):
    job = Job()
    st.session_state.job = job
    def worker():
        try:
            run_job(source, ROOT / 'data/output', config, job)
        finally:
            if temporary:
                Path(source).unlink(missing_ok=True)
    thread = Thread(target=worker, daemon=True)
    thread.start()

with detection:
    st.subheader('Detection and Tracking')
    upload = None
    if source_type == 'Upload video or image':
        upload = st.file_uploader('Choose a video or image', type=['mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png', 'bmp', 'webp'])
        if upload:
            with st.expander('Original input', expanded=False):
                if Path(upload.name).suffix.lower() in IMAGE_SUFFIXES:
                    st.image(upload.getvalue())
                    st.caption('Still images show detections only; tracking IDs and crossing counts require video.')
                else:
                    st.video(upload.getvalue())
            st.caption('Some source codecs cannot preview in browsers, but may still decode with OpenCV.')
    else:
        st.warning('This opens a camera physically connected to the Streamlit server. It does not access a remote browser camera. Use locally; uploads work on remote deployments.')
        camera = st.number_input('Local camera index', 0, 10, 0)
        seconds = st.slider('Maximum captured frames', 30, 3000, 300)
    running = 'job' in st.session_state and not st.session_state.job.snapshot()['done']
    if st.button('Start processing', type='primary', disabled=running or (source_type == 'Upload video or image' and upload is None)):
        try:
            config = ProcessingConfig(model=model, tracker=tracker, confidence=confidence, iou=iou,
                display_confidence=display_confidence,
                min_track_hits=min_track_hits, stabilize_classes=stabilize_classes, class_window=class_window,
                track_buffer=track_buffer, match_threshold=match_threshold, show_tentative=show_tentative, show_summary=show_summary,
                duplicate_iou=duplicate_iou,
                imgsz=imgsz, device=device, classes=selected, trail_length=trail_length,
                show_trails=trails, centroids=centroids, counting=counting,
                line=((x1, y1), (x2, y2)), tolerance=tolerance, stride=stride, max_width=max_width,
                max_frames=int(max_frames) if source_type == 'Upload video or image' else seconds)
            if upload:
                input_dir = ROOT / 'data/input'
                input_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=input_dir, suffix=Path(upload.name).suffix, delete=False) as handle:
                    handle.write(upload.getbuffer())
                    source = handle.name
                launch(source, config, temporary=True)
            else:
                launch(int(camera), config)
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    @st.fragment(run_every=1)
    def progress_panel():
        job = st.session_state.get('job')
        if not job:
            st.info('Your annotated preview will appear here.')
            return
        state = job.snapshot()
        st.write(f"{state['status']} · {state['processed']} processed frames")
        if state['total']:
            st.progress(min(1., state['processed'] / state['total']))
        if not state['done'] and st.button('Stop and save partial output'):
            job.cancel.set()
        if state['preview'] is not None:
            st.image(state['preview'], caption='Latest processed frame', width='stretch')
        if state['error']:
            st.error(state['error'])
        if state['done'] and st.session_state.get('finished_job') is not job:
            st.session_state.finished_job = job
            if state['result']:
                st.session_state.result = state['result']
            st.rerun()
    progress_panel()

result = st.session_state.get('result')
if result:
    folder = Path(result['folder'])
    summary = result['summary']
    records = pd.read_csv(folder / 'tracking.csv')
    frames = pd.read_csv(folder / 'frames.csv')
    counts = pd.read_csv(folder / 'counts.csv')
    with detection:
        st.subheader('Saved output')
        for warning in summary['warnings']:
            st.warning(warning)
        is_image = summary.get('media_type') == 'image'
        media = result['image'] if is_image else result['video']
        if is_image:
            st.image(media, caption='Detected objects (no temporal IDs for still images)')
        elif Path(result['video']).name == 'annotated.mp4':
            st.video(result['video'])
        downloads = [(Path(media).name, 'image/jpeg' if is_image else 'video/mp4'), ('tracking.csv', 'text/csv'), ('counts.csv', 'text/csv'), ('tracks.csv', 'text/csv'), ('frames.csv', 'text/csv'), ('analytics.json', 'application/json')]
        if (folder / 'observations.csv').exists():
            downloads.append(('observations.csv', 'text/csv'))
        for filename, mime in downloads:
            st.download_button(f'Download {filename}', (folder / filename).read_bytes(), filename, mime)
    with analytics:
        columns = st.columns(4)
        for col, label, value in zip(columns, ['Unique IDs', 'Visible in last frame', 'Line crossings', 'Processing FPS'],
            [summary['unique_objects'], summary['currently_visible_objects'], summary['total_crossings'], round(summary['processing_fps'], 2)]):
            col.metric(label, value)
        st.caption(f"{summary['frames_processed']} frames · {summary['processing_seconds']:.2f}s processing · {summary['average_inference_ms']:.2f} ms average detector/tracker call")
        st.caption('Unique IDs are observed track identities, not guaranteed physical individuals. FPS includes initialization but excludes final encoding/export. No ground-truth accuracy metrics are asserted.')
        if is_image:
            st.caption('A still image has no tracks or crossings. Counts below are image detections.')
            image_counts = pd.DataFrame(list(summary.get('image_counts', {}).items()), columns=['object_class', 'detections'])
            if not image_counts.empty:
                st.plotly_chart(px.bar(image_counts, x='object_class', y='detections', title='Image detections by category'), width='stretch')
        if not counts.empty and not is_image:
            st.plotly_chart(px.bar(counts, x='object_class', y='unique_objects', title='Unique track IDs by category'), width='stretch')
            directions = counts.melt(id_vars='object_class', value_vars=['a_to_b', 'b_to_a'], var_name='direction', value_name='crossings')
            st.plotly_chart(px.bar(directions, x='direction', y='crossings', color='object_class', barmode='group', title='Line crossings by direction'), width='stretch')
        st.plotly_chart(px.line(frames, x='timestamp', y=['visible_objects', 'tracked_objects'], title='Objects over video time'), width='stretch')
        st.plotly_chart(px.line(frames, x='frame_number', y='processing_fps', title='Frame processing FPS'), width='stretch')
        if not records.empty:
            st.plotly_chart(px.histogram(records, x='confidence', color='raw_class' if 'raw_class' in records else 'object_class', title='Raw detection confidence distribution'), width='stretch')
    with tracking:
        st.caption('Coordinates refer to the resized output. Timestamps use source frame time (wall time for webcam). Blank IDs are untracked detections.')
        st.dataframe(records, width='stretch')
        st.caption('object_class is the stabilized label; raw_class and confidence preserve the latest model output. class_support is a weighted vote share, not accuracy. observations.csv includes tentative and low-score observations, so filtering stays auditable.')
        st.subheader('Entry and last observation')
        st.dataframe(pd.read_csv(folder / 'tracks.csv'), width='stretch')
        st.caption('Last observation is not proof of a physical exit; occlusion or end of recording can terminate a track.')
else:
    with analytics:
        st.info('Process a video to populate analytics.')
    with tracking:
        st.info('Tracking records will appear after a run.')

with info:
    st.markdown((ROOT / 'docs/architecture.md').read_text(encoding='utf-8'))
    with st.expander('Algorithms and limitations'):
        st.markdown((ROOT / 'docs/algorithm.md').read_text(encoding='utf-8'))
    st.subheader('Processing history on this server')
    st.dataframe(Database(ROOT / 'data/output/history.db').recent(), width='stretch')
