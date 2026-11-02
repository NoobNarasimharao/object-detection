# Viva questions and answers

1. **What is object detection?** Identifying object categories and localizing each instance, usually with bounding boxes.
2. **What is object tracking?** Associating observations across frames to maintain a temporal identity and motion history.
3. **Detection versus classification?** Classification predicts a category for an image or region; detection also locates multiple instances.
4. **What is YOLO?** A family of single-stage detectors that predict boxes and class scores in a forward pass.
5. **How does YOLO work?** A backbone extracts features, multi-scale feature aggregation combines information, and prediction heads estimate object geometry and classes.
6. **Why use pretrained weights?** They provide learned visual representations and immediate supported-class detection without building a new labeled training dataset.
7. **What is ByteTrack?** A multi-object tracker that associates high-confidence detections first and then uses lower-confidence detections to recover unmatched tracks.
8. **How are detections associated?** Motion predictions and overlap-based costs form a gated linear assignment problem between tracks and detections.
9. **What is IoU?** Intersection area divided by union area for two boxes; it measures spatial overlap from zero to one.
10. **What is a Kalman filter?** A recursive state estimator combining motion predictions with noisy measurements and uncertainty estimates.
11. **What is the Hungarian algorithm?** An algorithm solving minimum-cost one-to-one assignment. The library may use another efficient solver for the same assignment problem.
12. **Who assigns IDs?** Ultralytics' tracker assigns IDs when tracks are initialized. The application only extracts them.
13. **What happens during occlusion?** A track can persist in a lost state and later associate again, but long occlusion can produce a new ID.
14. **What causes ID switches?** Ambiguous overlap, inaccurate motion estimates, missed detections, similar nearby objects, and abrupt camera/object motion.
15. **What is FPS?** Frames processed per second. Its value depends on whether timing includes decoding, rendering, model loading and export.
16. **How is GPU inference selected?** PyTorch checks CUDA availability; auto selects GPU 0 when available and CPU otherwise.
17. **ByteTrack versus DeepSORT?** ByteTrack emphasizes two-stage high/low-score association; DeepSORT adds learned appearance descriptors to motion-based association.
18. **Why combine YOLO and ByteTrack?** YOLO supplies fast semantic detections; ByteTrack adds temporal identity using relatively lightweight association.
19. **What are major limitations?** Missed detections, ID changes, domain shift, low light, small objects, crowded scenes and hardware-dependent speed.
20. **Does confidence equal accuracy?** No. It is a detector score; actual accuracy requires labeled ground truth and a defined metric.
21. **Why keep a low detector threshold?** Filtering out low-score candidates before tracking prevents ByteTrack's second stage from using them.
22. **How is a centroid calculated?** Average the left/right x coordinates and top/bottom y coordinates.
23. **How are repeated crossings prevented?** A dead band ignores near-line jitter; a set permits each ID at most one crossing per direction.
24. **Why check a finite line segment?** An object crossing the infinite extension outside the user-drawn endpoints should not count.
25. **Unique count versus visible count?** Unique count accumulates distinct IDs; visible count describes detections in one frame.
26. **What does persist=True do?** It tells tracking that successive supplied frames belong to the same sequence, preserving association state.
27. **Why reset between videos?** Otherwise state and counters from one sequence could contaminate an unrelated sequence.
28. **How does frame skipping affect tracking?** It increases motion between observations and can break association; the track buffer effectively spans more source time.
29. **Why is H.264 conversion useful?** Browser players commonly support H.264 MP4 more reliably than OpenCV's MPEG-4 Part 2 output.
30. **Can OpenCV access a remote visitor's webcam?** No; VideoCapture opens a device attached to the server. Remote browser video needs a separate client streaming mechanism.
31. **What does observation duration mean?** Time from first to last observation, including gaps; it does not prove continuous visibility.
32. **What ground truth is needed for MOTA/IDF1?** Frame-by-frame object identities and locations, plus a matching protocol; confidence scores alone cannot produce those metrics.
33. **Why use nms=None here?** It chooses YOLO26's NMS inference path so the UI IoU threshold controls suppression.
34. **What does BoT-SORT add?** Camera-motion compensation and support for optional appearance re-identification, with additional complexity.
35. **How is the UI kept responsive?** Processing runs in an isolated thread; a Streamlit fragment polls progress and sets a cancellation event.
