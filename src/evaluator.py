import numpy as np

def evaluate(frames, confidence, elapsed):
    inference = [f['inference_ms'] for f in frames]
    return dict(frames_processed=len(frames), processing_seconds=elapsed,
                processing_fps=len(frames) / elapsed if elapsed else 0,
                average_inference_ms=float(np.mean(inference)) if inference else 0,
                confidence_mean=float(np.mean(confidence)) if confidence else None,
                confidence_min=float(min(confidence)) if confidence else None,
                confidence_max=float(max(confidence)) if confidence else None)
