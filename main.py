import argparse
from dataclasses import replace
import json
from src.video_processor import ProcessingConfig, VideoProcessor
from utils.helpers import defaults

def main():
    d = defaults()
    parser = argparse.ArgumentParser(description='YOLO26 + ByteTrack video laboratory')
    parser.add_argument('--source', required=True, help='Video/image path, or webcam:0 for a local camera')
    parser.add_argument('--model', default=d['model'])
    parser.add_argument('--tracker', choices=['bytetrack.yaml', 'botsort.yaml'], default=d['tracker'])
    parser.add_argument('--confidence', type=float, default=d['confidence'])
    parser.add_argument('--display-confidence', type=float, default=d['display_confidence'],
                        help='Evidence threshold for track confirmation; image detection threshold')
    parser.add_argument('--iou', type=float, default=d['iou'])
    parser.add_argument('--device', default=d['device'])
    parser.add_argument('--classes', type=int, nargs='+', help='COCO class IDs, e.g. 0 2 3 5 7')
    parser.add_argument('--output', default='data/output')
    parser.add_argument('--imgsz', type=int, default=d['imgsz'])
    parser.add_argument('--max-width', type=int, default=d['max_width'], help='Output width limit only; 0 preserves source size')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--trail-length', type=int, default=d['trail_length'])
    parser.add_argument('--show-trajectories', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--counting', action=argparse.BooleanOptionalAction, default=d['counting'])
    parser.add_argument('--min-track-hits', type=int, default=d['min_track_hits'])
    parser.add_argument('--class-window', type=int, default=d['class_window'])
    parser.add_argument('--track-buffer', type=int, default=d['track_buffer'])
    parser.add_argument('--match-threshold', type=float, default=.8)
    parser.add_argument('--duplicate-iou', type=float, default=.95, help='Near-identical box suppression; 0 disables')
    parser.add_argument('--stabilize-classes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--show-tentative', action='store_true')
    parser.add_argument('--show-summary', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--line', type=float, nargs=4, default=[.1, .5, .9, .5], metavar=('X1','Y1','X2','Y2'))
    parser.add_argument('--save', action='store_true', help='Outputs are always saved; accepted for compatibility')
    parser.add_argument('--compare', action='store_true', help='Run both trackers independently on the same video')
    args = parser.parse_args()
    try:
        source = int(args.source.split(':')[1]) if args.source.startswith('webcam:') else args.source
        if args.compare and isinstance(source, int):
            parser.error('Comparison requires a prerecorded video.')
        config = ProcessingConfig(model=args.model, tracker=args.tracker, confidence=args.confidence,
            display_confidence=args.display_confidence,
            min_track_hits=args.min_track_hits, class_window=args.class_window, track_buffer=args.track_buffer,
            match_threshold=args.match_threshold, stabilize_classes=args.stabilize_classes,
            duplicate_iou=args.duplicate_iou,
            show_tentative=args.show_tentative, show_summary=args.show_summary,
            iou=args.iou, device=args.device, classes=args.classes, imgsz=args.imgsz, max_width=args.max_width,
            stride=args.stride, max_frames=args.max_frames, trail_length=args.trail_length,
            show_trails=args.show_trajectories, counting=args.counting, line=(tuple(args.line[:2]), tuple(args.line[2:])))
        for tracker in (['bytetrack.yaml', 'botsort.yaml'] if args.compare else [args.tracker]):
            result = VideoProcessor(replace(config, tracker=tracker)).process(source, args.output)
            print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'Processing failed: {exc}\n')

if __name__ == '__main__':
    main()
