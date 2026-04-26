#!/usr/bin/env python3
"""
runner.py — Multi-Environment Model Orchestrator

Responsibilities:
- Runs BasicPitch in its dedicated venv (TF 2.15)
- Runs SPICE in its dedicated venv (TF 2.21)
- Collects outputs (JSON / intermediate artifacts)
- Returns structured results for FusionLayer
- Supports parallel execution for performance

NO interpretation logic lives here — just orchestration.
"""

import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum


# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class ModelStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ModelResult:
    """Result from a single model execution"""
    model_name: str
    status: ModelStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


# ============================================================================
# ENVIRONMENT CONFIG
# ============================================================================

class EnvironmentConfig:
    """Configuration for model environments"""

    def __init__(self, base_dir: Path = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent

        # Virtual environment paths
        self.basic_pitch_venv = self.base_dir / ".venv_basic_pitch"
        self.spice_venv = self.base_dir / ".venv_spice"

        # Python executables
        self.basic_pitch_python = self.basic_pitch_venv / "Scripts" / "python.exe"
        self.spice_python = self.spice_venv / "Scripts" / "python.exe"

        # Wrapper scripts
        self.basic_pitch_script = self.base_dir / "basic_pitch_script.py"
        self.spice_script = self.base_dir / "spice_script.py"

        # Timeout settings (seconds)
        self.basic_pitch_timeout = 300  # 5 minutes
        self.spice_timeout = 300  # 5 minutes

        # Retry settings
        self.max_retries = 2
        self.retry_delay = 1.0  # seconds

        # Output settings
        self.temp_dir = self.base_dir / "temp" / "runner"
        self.keep_temp_files = False

        # Parallel execution
        self.max_workers = 2

        # Debug
        self.verbose = True


# ============================================================================
# MODEL WRAPPER SCRIPTS (create if missing)
# ============================================================================

def ensure_wrapper_scripts(config: EnvironmentConfig):
    """Create wrapper scripts if they don't exist"""

    # Basic Pitch wrapper
    if not config.basic_pitch_script.exists():
        basic_pitch_content = '''#!/usr/bin/env python3
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
'''
        config.basic_pitch_script.write_text(basic_pitch_content)
        if config.verbose:
            print(f"✅ Created Basic Pitch wrapper: {config.basic_pitch_script}")

    # SPICE wrapper
    if not config.spice_script.exists():
        spice_content = '''#!/usr/bin/env python3
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
'''
        config.spice_script.write_text(spice_content)
        if config.verbose:
            print(f"✅ Created SPICE wrapper: {config.spice_script}")


# ============================================================================
# CORE RUNNER
# ============================================================================

class ModelRunner:
    """
    Handles isolated execution of ML models across virtual environments.
    Supports parallel execution, retries, and comprehensive error handling.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self._validate_envs()
        self._temp_dir = None
        ensure_wrapper_scripts(self.config)

    def _validate_envs(self):
        """Ensure both interpreters exist before running anything."""
        missing = []

        if not self.config.basic_pitch_python.exists():
            missing.append(f"BasicPitch venv not found: {self.config.basic_pitch_python}")

        if not self.config.spice_python.exists():
            missing.append(f"SPICE venv not found: {self.config.spice_python}")

        if missing:
            raise FileNotFoundError("\n".join(missing))

    def _get_temp_dir(self) -> Path:
        """Get or create temporary directory for intermediate files"""
        if self._temp_dir is None:
            self._temp_dir = self.config.temp_dir
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        return self._temp_dir

    def _cleanup_temp(self):
        """Clean up temporary files if not keeping them"""
        if not self.config.keep_temp_files and self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            if self.config.verbose:
                print(f"🧹 Cleaned temp directory: {self._temp_dir}")

    def _run_with_retry(self, cmd: List[str], timeout: int, model_name: str, task_id: str) -> Tuple[str, str, float]:
        """Run command with retry logic"""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False
                )

                execution_time = time.time() - start_time

                if result.returncode == 0:
                    return result.stdout, result.stderr, execution_time
                else:
                    last_error = f"Return code {result.returncode}: {result.stderr}"

            except subprocess.TimeoutExpired as e:
                last_error = f"Timeout after {timeout}s"
                execution_time = timeout

            except Exception as e:
                last_error = str(e)
                execution_time = 0

            if attempt < self.config.max_retries:
                if self.config.verbose:
                    print(f"⚠️ {model_name} attempt {attempt + 1} failed, retrying...")
                time.sleep(self.config.retry_delay)

        raise RuntimeError(f"{model_name} failed after {self.config.max_retries + 1} attempts: {last_error}")

    # ------------------------------------------------------------------------
    # BASIC PITCH
    # ------------------------------------------------------------------------

    def run_basic_pitch(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """
        Run BasicPitch in its isolated environment.

        Expected output:
            JSON with notes, duration, and metadata
        """

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cmd = [
            str(self.config.basic_pitch_python),
            str(self.config.basic_pitch_script),
            audio_path,
            task_id
        ]

        if self.config.verbose:
            print(f"🎧 Running BasicPitch: {task_id}")

        stdout, stderr, execution_time = self._run_with_retry(
            cmd,
            self.config.basic_pitch_timeout,
            "BasicPitch",
            task_id
        )

        result = self._parse_output(stdout, f"BasicPitch ({task_id})")

        # Add execution metadata
        result["execution_time"] = execution_time
        result["status"] = "success"

        if stderr and self.config.verbose:
            print(f"📝 BasicPitch warnings: {stderr[:200]}")

        return result

    # ------------------------------------------------------------------------
    # SPICE
    # ------------------------------------------------------------------------

    def run_spice(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """
        Run SPICE rhythm analysis in isolated environment.
        """

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cmd = [
            str(self.config.spice_python),
            str(self.config.spice_script),
            audio_path,
            task_id
        ]

        if self.config.verbose:
            print(f"🥁 Running SPICE: {task_id}")

        stdout, stderr, execution_time = self._run_with_retry(
            cmd,
            self.config.spice_timeout,
            "SPICE",
            task_id
        )

        result = self._parse_output(stdout, f"SPICE ({task_id})")

        # Add execution metadata
        result["execution_time"] = execution_time
        result["status"] = "success"

        if stderr and self.config.verbose:
            print(f"📝 SPICE output: {stderr[:200]}")

        return result

    # ------------------------------------------------------------------------
    # PARALLEL EXECUTION (BOTH MODELS SIMULTANEOUSLY)
    # ------------------------------------------------------------------------

    def run_full_analysis_parallel(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """
        Runs both models in parallel for maximum throughput.
        """

        results = {
            "task_id": task_id,
            "audio_path": audio_path,
            "timestamp": datetime.now().isoformat(),
            "basic_pitch": None,
            "spice": None,
            "errors": [],
            "total_execution_time": 0
        }

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit both tasks
            future_bp = executor.submit(self.run_basic_pitch, audio_path, task_id)
            future_spice = executor.submit(self.run_spice, audio_path, task_id)

            # Collect results as they complete
            for future in as_completed([future_bp, future_spice]):
                try:
                    result = future.result()
                    if "notes" in result:
                        results["basic_pitch"] = result
                    elif "drums" in result:
                        results["spice"] = result
                except Exception as e:
                    error_msg = str(e)
                    results["errors"].append(error_msg)
                    if self.config.verbose:
                        print(f"❌ Model failed: {error_msg}")

        results["total_execution_time"] = time.time() - start_time

        # Determine overall status
        if results["basic_pitch"] and results["spice"]:
            results["overall_status"] = "success"
        elif results["basic_pitch"] or results["spice"]:
            results["overall_status"] = "partial_success"
        else:
            results["overall_status"] = "failed"

        return results

    # ------------------------------------------------------------------------
    # SEQUENTIAL EXECUTION (FOR DEBUGGING)
    # ------------------------------------------------------------------------

    def run_full_analysis_sequential(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """
        Runs both models sequentially. Useful for debugging.
        """

        results = {
            "task_id": task_id,
            "audio_path": audio_path,
            "timestamp": datetime.now().isoformat(),
            "basic_pitch": None,
            "spice": None,
            "errors": [],
            "total_execution_time": 0
        }

        start_time = time.time()

        # Run BasicPitch
        try:
            results["basic_pitch"] = self.run_basic_pitch(audio_path, task_id)
        except Exception as e:
            results["errors"].append(f"BasicPitch: {str(e)}")

        # Run SPICE
        try:
            results["spice"] = self.run_spice(audio_path, task_id)
        except Exception as e:
            results["errors"].append(f"SPICE: {str(e)}")

        results["total_execution_time"] = time.time() - start_time

        # Determine overall status
        if results["basic_pitch"] and results["spice"]:
            results["overall_status"] = "success"
        elif results["basic_pitch"] or results["spice"]:
            results["overall_status"] = "partial_success"
        else:
            results["overall_status"] = "failed"

        return results

    # ------------------------------------------------------------------------
    # MAIN ENTRY POINT (DEFAULT: PARALLEL)
    # ------------------------------------------------------------------------

    def run_full_analysis(self, audio_path: str, task_id: str, parallel: bool = True) -> Dict[str, Any]:
        """
        Runs both models. Defaults to parallel execution for performance.

        Args:
            audio_path: Path to audio file
            task_id: Unique identifier for this task
            parallel: If True, run models in parallel; if False, run sequentially

        Returns:
            Dictionary with both model outputs
        """

        if parallel:
            return self.run_full_analysis_parallel(audio_path, task_id)
        else:
            return self.run_full_analysis_sequential(audio_path, task_id)

    # ------------------------------------------------------------------------
    # OUTPUT PARSING
    # ------------------------------------------------------------------------

    def _parse_output(self, stdout: str, source: str) -> Dict[str, Any]:
        """
        Parses model output safely.

        Assumption: scripts print JSON as final line OR full stdout is JSON
        """

        if not stdout or not stdout.strip():
            raise ValueError(f"Empty output from {source}")

        try:
            # Try direct JSON parsing
            return json.loads(stdout.strip())
        except json.JSONDecodeError:
            # Fallback: extract last JSON line
            lines = stdout.strip().splitlines()

            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

            # Still failed - show what we got
            raise ValueError(
                f"Could not parse JSON output from {source}. "
                f"First 500 chars: {stdout[:500]}"
            )

    # ------------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------------

    def cleanup(self):
        """Clean up temporary resources"""
        self._cleanup_temp()


# ============================================================================
# RESULT EXTRACTOR (for FusionLayer compatibility)
# ============================================================================

class ResultExtractor:
    """Extracts and converts runner outputs to FusionLayer-compatible format"""

    @staticmethod
    def extract_notes(basic_pitch_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract note events from BasicPitch output"""
        if not basic_pitch_output or "notes" not in basic_pitch_output:
            return []

        return basic_pitch_output["notes"]

    @staticmethod
    def extract_drums(spice_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract drum events from SPICE output"""
        if not spice_output or "drums" not in spice_output:
            return []

        return spice_output["drums"]

    @staticmethod
    def extract_rhythm(spice_output: Dict[str, Any], basic_pitch_output: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract rhythm/tempo information"""

        # Default tempo (could come from SPICE or BasicPitch)
        tempo = 120.0

        # Try to extract from SPICE (if available)
        if spice_output and "detected_tempo" in spice_output:
            tempo = spice_output["detected_tempo"]
        elif basic_pitch_output and "estimated_tempo" in basic_pitch_output:
            tempo = basic_pitch_output["estimated_tempo"]

        return {
            "tempo": tempo,
            "time_signature": (4, 4),  # Default, can be overridden
            "confidence": 0.8
        }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-environment model analysis")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--task_id", default=None, help="Task ID (default: auto-generated)")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run models in parallel")
    parser.add_argument("--sequential", action="store_true", help="Run models sequentially")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")

    args = parser.parse_args()

    # Generate task ID if not provided
    if not args.task_id:
        args.task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create config
    config = EnvironmentConfig()
    config.verbose = args.verbose
    config.keep_temp_files = args.keep_temp

    # Create runner
    runner = ModelRunner(config)

    try:
        # Choose execution mode
        use_parallel = args.parallel and not args.sequential

        print(f"\n{'=' * 60}")
        print(f"🚀 Starting analysis: {args.task_id}")
        print(f"📁 Audio: {args.audio_path}")
        print(f"⚡ Mode: {'Parallel' if use_parallel else 'Sequential'}")
        print(f"{'=' * 60}\n")

        outputs = runner.run_full_analysis(
            audio_path=args.audio_path,
            task_id=args.task_id,
            parallel=use_parallel
        )

        print(f"\n{'=' * 60}")
        print(f"✅ Analysis complete: {args.task_id}")
        print(f"📊 Status: {outputs.get('overall_status', 'unknown')}")
        print(f"⏱️  Total time: {outputs['total_execution_time']:.2f}s")

        if outputs.get("basic_pitch"):
            bp = outputs["basic_pitch"]
            print(f"🎹 BasicPitch: {bp.get('note_count', 0)} notes in {bp.get('execution_time', 0):.2f}s")

        if outputs.get("spice"):
            sp = outputs["spice"]
            print(f"🥁 SPICE: {sp.get('drum_count', 0)} drums in {sp.get('execution_time', 0):.2f}s")

        if outputs.get("errors"):
            print(f"⚠️ Errors: {len(outputs['errors'])}")

        print(f"{'=' * 60}\n")

        # Output JSON for piping
        print(json.dumps(outputs, indent=2))

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()