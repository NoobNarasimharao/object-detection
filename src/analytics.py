import json
import pandas as pd

RECORD_COLUMNS = ['track_id', 'object_class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'centroid_x', 'centroid_y', 'frame_number', 'timestamp', 'raw_class', 'raw_class_id', 'class_support', 'confirmed', 'duplicate_of']
FRAME_COLUMNS = ['frame_number', 'timestamp', 'visible_objects', 'tracked_objects', 'inference_ms', 'processing_fps']

def export_results(folder, records, frames, counts, tracks, summary, observations=None):
    pd.DataFrame(records, columns=RECORD_COLUMNS).to_csv(folder / 'tracking.csv', index=False)
    pd.DataFrame(frames, columns=FRAME_COLUMNS).to_csv(folder / 'frames.csv', index=False)
    pd.DataFrame(counts, columns=['object_class', 'unique_objects', 'a_to_b', 'b_to_a', 'image_detections']).to_csv(folder / 'counts.csv', index=False)
    pd.DataFrame(tracks, columns=['track_id', 'object_class', 'entry_frame', 'entry_time', 'last_seen_frame', 'last_seen_time', 'observations', 'duration_seconds']).to_csv(folder / 'tracks.csv', index=False)
    (folder / 'analytics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if observations is not None:
        pd.DataFrame(observations, columns=RECORD_COLUMNS).to_csv(folder / 'observations.csv', index=False)
