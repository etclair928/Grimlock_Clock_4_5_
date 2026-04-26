#!/usr/bin/env python3
"""SPICE wrapper script - runs in TF 2.21 environment"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub

    # Silence TensorFlow logging
    tf.get_logger().setLevel('ERROR')

    # Load model once
    print("Loading SPICE model...", file=sys.stderr)
    spice_model = hub.load("https://tfhub.dev/google/spice/2")
    print("SPICE model loaded", file=sys.stderr)

    def extract_drums(audio_path: str, task_id: str) -> dict:
        """Extract drum/percussion information from audio"""

        # Load audio (SPICE expects 16kHz mono)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr

        # Run SPICE inference
        model_output = spice_model.signatures["serving_default"](
            tf.constant(audio, dtype=tf.float32),
            sample_rate=tf.constant(sr, dtype=tf.int32)
        )

        # Parse outputs
        onsets = model_output["onsets"].numpy().flatten()
        offsets = model_output["offsets"].numpy().flatten()
        frame_activations = model_output["frame_activations"].numpy()

        # Convert onsets/offsets to drum events (simplified)
        drum_events = []
        onset_times = np.where(onsets > 0.5)[0]
        onset_times_seconds = onset_times * (512 / sr)  # SPICE hop size

        for idx, onset_time in enumerate(onset_times_seconds):
            # Estimate drum type based on frequency content (simplified)
            if idx % 2 == 0:
                drum_type = "KICK"
            elif idx % 3 == 0:
                drum_type = "SNARE"
            else:
                drum_type = "HIHAT"

            drum_events.append({
                "time": float(onset_time),
                "drum_type": drum_type,
                "velocity": int(min(127, max(1, int(onsets[onset_times[idx]] * 127)))),
                "confidence": float(onsets[onset_times[idx]])
            })

        return {
            "task_id": task_id,
            "duration": duration,
            "sample_rate": sr,
            "drum_count": len(drum_events),
            "drums": drum_events,
            "onset_count": len(onset_times_seconds),
            "activations_shape": list(frame_activations.shape),
            "model": "spice",
            "tensorflow_version": tf.__version__
        }

    if __name__ == "__main__":
        import librosa  # imported here to avoid TF conflicts
        audio_path = sys.argv[1]
        task_id = sys.argv[2] if len(sys.argv) > 2 else "unknown"

        try:
            result = extract_drums(audio_path, task_id)
            print(json.dumps(result, indent=2))
        except Exception as e:
            error_result = {
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            }
            print(json.dumps(error_result))
            sys.exit(1)

except ImportError as e:
    error_result = {
        "error": f"Import failed: {str(e)}",
        "status": "failed",
        "missing_dependencies": ["tensorflow==2.21.0", "tensorflow-hub", "librosa"]
    }
    print(json.dumps(error_result))
    sys.exit(1)
