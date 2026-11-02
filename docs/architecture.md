# System architecture

This application separates frame processing from presentation. Streamlit and the CLI use the same `VideoProcessor`, which owns capture/writer resources, one detector instance and one set of run statistics.

```text
Image / video upload / server camera / CLI file
             ↓
OpenCV capture → optional frame sampling (original pixels)
             ↓
YOLO26 PyTorch detection → explicit ByteTrack association
             ↓
Tracker ID extraction → confirmation + class voting → output resize → histories/counters
             ↓
OpenCV annotations → MP4 writer → FFmpeg H.264
             ↓
CSV + JSON + SQLite → Streamlit tables and Plotly charts
```

The worker thread never calls Streamlit APIs. A lock protects progress snapshots; an Event requests cancellation. A Streamlit fragment polls the snapshot while the main interface remains responsive. Every Start creates a separate worker, config and model. A session cannot start a second worker while its first is active. A rerun displays state without restarting processing.

`detector.py` translates Ultralytics results into typed detections. `tracker.py` maintains bounded trails and first/last observations; association itself remains inside Ultralytics ByteTrack. `counter.py` applies geometric crossing rules. `visualizer.py` renders. `analytics.py` exports, `evaluator.py` computes measured operational metrics, and `database.py` stores history transactionally.

Python, PyTorch, Ultralytics, OpenCV and NumPy provide inference and geometry; Pandas, Plotly, Streamlit and SQLite provide analysis and presentation. CUDA is selected automatically only if PyTorch reports it available. All paths are derived from the repository or explicit CLI arguments.

Normalized line endpoints are converted to the resized output dimensions. Exported coordinates use the same dimensions as rendered boxes. Timestamp and frame identity are preserved from the source even when processing every Nth frame.

Memory grows with exported detections and the number of unique tracks, although each visual trail is bounded. Video writing is streamed to disk. The laboratory application is designed for bounded clips, not unbounded surveillance retention.

Images take a separate single-frame predict path with no temporal IDs. `observations.csv` preserves pre-confirmation evidence. Every video run writes the exact ByteTrack/BoT-SORT YAML, including configurable lost-track buffer and assignment threshold. Raw and stabilized labels are both exported. Class corrections migrate unique/crossing totals between classes.
