#!/usr/bin/env python3
"""Basic Pitch wrapper script - runs in TF 2.15 environment"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")


try:
    import librosa
    import numpy as np
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    def extract_pitch(audio_path: str, task_id: str) -> dict:
        """Extract pitch/melody information from audio"""

        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(audio) / sr

        # Run prediction
        model_output, midi_data, note_events = predict(
            audio_path,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,  # in ms
            minimum_frequency=80,
            maximum_frequency=2000
        )

        # Convert to serializable format
        notes = []
        for note in note_events:
            notes.append({
                "pitch": int(note.pitch),
                "start": float(note.start),
                "end": float(note.end),
                "velocity": int(note.velocity),
                "confidence": float(note.confidence) if hasattr(note, 'confidence') else 0.8
            })

        return {
            "task_id": task_id,
            "duration": duration,
            "sample_rate": sr,
            "note_count": len(notes),
            "notes": notes,
            "model": "basic-pitch",
            "tensorflow_version": "2.15.0"
        }

    if __name__ == "__main__":
        audio_path = sys.argv[1]
        task_id = sys.argv[2] if len(sys.argv) > 2 else "unknown"

        try:
            result = extract_pitch(audio_path, task_id)
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
        "missing_dependencies": ["basic-pitch", "librosa", "tensorflow==2.15.0"]
    }
    print(json.dumps(error_result))
    sys.exit(1)
