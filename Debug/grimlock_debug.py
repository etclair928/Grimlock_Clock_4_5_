#!/usr/bin/env python3
r"""
grimlock_debug.py — Complete Pipeline Diagnostic Tool

Usage:
    python grimlock_debug.py [--audio <path>] [--truncate <seconds>]

Default audio: C:\Users\kiwi2\Downloads\Hopeful.mp3

This script traces EVERY step of the pipeline and shows:
    - Which modules are loading
    - Parameter passing issues
    - Where data is being lost
    - Actual vs placeholder notes
    - Tempo detection results
"""

import os
import sys
import asyncio
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Default audio path - using forward slashes to avoid escape issues
DEFAULT_AUDIO = Path(r"C:\Users\kiwi2\Downloads\Hopeful.mp3")


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_section(title: str):
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")


def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.ENDC}")


def print_info(msg: str):
    print(f"{Colors.BLUE}📌 {msg}{Colors.ENDC}")


def print_value(label: str, value: Any):
    print(f"   {Colors.DIM}{label}:{Colors.ENDC} {value}")


# ============================================================================
# MODULE IMPORT TESTER
# ============================================================================

def test_imports():
    """Test all critical imports and report which fail."""
    print_section("1. MODULE IMPORT TESTS")

    modules_to_test = {
        "order_types": "from order_types import Note, DrumHit, RhythmInfo, TranscriptionResult",
        "grimlock_pipeline": "from grimlock_pipeline import GrimlockPipeline",
        "engine.state_manager": "from engine.state_manager import StateManager",
        "modules.rhythm_engine": "from modules.rhythm_engine import RhythmEngine",
        "modules.drum_intelligence": "from modules.drum_intelligence import DrumIntelligence",
        "modules.pitch_intelligence": "from modules.pitch_intelligence import PitchIntelligence",
        "engine.fusion_layer": "from engine.fusion_layer import GrimlockFusionLayer",
        "separation.hybrid_separator": "from separation.hybrid_separator import HybridSeparator",
    }

    results = {}
    for name, import_stmt in modules_to_test.items():
        try:
            exec(import_stmt)
            results[name] = True
            print_success(f"{name}")
        except ImportError as e:
            results[name] = False
            print_error(f"{name}: {e}")
        except Exception as e:
            results[name] = False
            print_error(f"{name}: {e}")

    return results


# ============================================================================
# METHOD SIGNATURE INSPECTOR
# ============================================================================

def inspect_method_signatures():
    """Inspect key method signatures to find parameter mismatches."""
    print_section("2. METHOD SIGNATURE INSPECTION")

    imports_to_try = [
        ("modules.pitch_intelligence", "PitchIntelligence"),
        ("modules.rhythm_engine", "RhythmEngine"),
        ("modules.drum_intelligence", "DrumIntelligence"),
        ("engine.fusion_layer", "GrimlockFusionLayer"),
    ]

    signatures = {}

    for module_name, class_name in imports_to_try:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Check process_piano signature
            if hasattr(cls, 'process_piano'):
                import inspect
                sig = inspect.signature(cls.process_piano)
                signatures[f"{class_name}.process_piano"] = str(sig)
                print_info(f"{class_name}.process_piano()")
                print_value("Signature", str(sig))

                # Check for 'state' parameter
                if 'state' in sig.parameters:
                    print_success("  ✅ Has 'state' parameter")
                else:
                    print_error("  ❌ MISSING 'state' parameter!")

            # Check process signature
            if hasattr(cls, 'process'):
                import inspect
                sig = inspect.signature(cls.process)
                signatures[f"{class_name}.process"] = str(sig)
                print_info(f"{class_name}.process()")
                print_value("Signature", str(sig))

                if 'state' in sig.parameters:
                    print_success("  ✅ Has 'state' parameter")
                else:
                    print_warning("  No 'state' parameter (may be optional)")

        except ImportError as e:
            print_error(f"Could not import {class_name}: {e}")
        except Exception as e:
            print_error(f"Error inspecting {class_name}: {e}")

    return signatures


# ============================================================================
# AUDIO FILE CHECKER
# ============================================================================

def check_audio_file(audio_path: Path) -> Path:
    """Check if audio file exists and is valid."""
    print_section("0. AUDIO FILE CHECK")

    print_value("Expected path", audio_path)

    if not audio_path.exists():
        print_error(f"Audio file NOT FOUND at: {audio_path}")

        # Try to find alternative
        alt_paths = [
            Path(r"C:\Users\kiwi2\Downloads\Hopeful.mp3"),
            Path(r"C:\Users\kiwi2\Downloads\test.mp3"),
            Path(r"C:\Users\kiwi2\Desktop\test.mp3"),
        ]

        for alt in alt_paths:
            if alt.exists():
                print_success(f"Found alternative: {alt}")
                return alt

        print_error("No audio file found! Please provide a valid path.")
        return None

    # File exists, check if it's readable
    try:
        import librosa
        import soundfile as sf
        import numpy as np

        # Quick test load
        y, sr = librosa.load(str(audio_path), duration=1.0, sr=None)
        print_success(f"Audio file is valid")
        print_value("Duration (first second test)", f"{len(y) / sr:.1f}s")
        print_value("Sample rate", f"{sr} Hz")
        print_value("File size", f"{audio_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Check if silent
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < 0.001:
            print_warning("Audio appears to be VERY quiet or silent")
        else:
            print_success(f"Audio level: RMS={rms:.4f}")

        return audio_path

    except Exception as e:
        print_error(f"Cannot read audio file: {e}")
        return None


# ============================================================================
# PIPELINE STEP DEBUGGER
# ============================================================================

class PipelineDebugger:
    """Trace every step of the pipeline with detailed logging."""

    def __init__(self, audio_path: Path, truncate_seconds: int = 30):
        self.audio_path = audio_path
        self.truncate_seconds = truncate_seconds
        self.steps = []
        self.audio = None
        self.sr = None
        self.stems = None
        self.rhythm_info = None
        self.drum_hits = None
        self.notes = None
        self.result = None
        self._np = None  # Will import numpy when needed

    @property
    def np(self):
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np

    def log_step(self, name: str, status: str, details: Any = None, duration: float = None):
        self.steps.append({
            "name": name,
            "status": status,
            "details": str(details)[:500] if details else None,
            "duration": duration
        })

        if status == "success":
            print_success(f"{name} ({duration:.2f}s)" if duration else f"{name}")
        elif status == "warning":
            print_warning(f"{name} ({duration:.2f}s)" if duration else f"{name}")
        else:
            print_error(f"{name} FAILED: {details}")

    async def test_audio_loading(self):
        """Test audio loading independently."""
        try:
            import librosa
            print_info("Loading audio...")
            start = time.time()

            duration_limit = self.truncate_seconds if self.truncate_seconds > 0 else None
            self.audio, self.sr = librosa.load(
                str(self.audio_path),
                sr=44100,
                duration=duration_limit
            )

            elapsed = time.time() - start
            print_value("Duration", f"{len(self.audio) / self.sr:.1f}s")
            print_value("Sample rate", f"{self.sr} Hz")
            print_value("Array shape", self.audio.shape)
            print_value("Array dtype", self.audio.dtype)
            print_value("RMS energy", f"{float(self.np.sqrt(self.np.mean(self.audio ** 2))):.6f}")

            self.log_step("Audio Loading", "success", duration=elapsed)
            return True

        except Exception as e:
            self.log_step("Audio Loading", "failed", str(e))
            print_error(f"Audio loading failed: {e}")
            traceback.print_exc()
            return False

    async def test_rhythm_engine(self):
        """Test rhythm engine with parameter inspection."""
        try:
            from modules.rhythm_engine import RhythmEngine

            print_info("Testing RhythmEngine...")
            start = time.time()

            # Create stems dict with drums
            stems = {'drums': self.audio if self.audio is not None else self.np.zeros(1)}

            # Try different ways to call
            engine = RhythmEngine()

            # Method 1: With state
            try:
                result = await engine.process(
                    stems=stems,
                    sr=self.sr,
                    state=None,  # Test with explicit None
                    duration=len(self.audio) / self.sr
                )
                print_success("RhythmEngine works with state=None")
                self.rhythm_info = result

            except TypeError as e:
                print_warning(f"State parameter issue: {e}")
                # Method 2: Without state
                try:
                    result = await engine.process(
                        stems=stems,
                        sr=self.sr,
                        duration=len(self.audio) / self.sr
                    )
                    print_success("RhythmEngine works without state")
                    self.rhythm_info = result
                except Exception as e2:
                    raise e2

            elapsed = time.time() - start
            if self.rhythm_info:
                print_value("Detected tempo", f"{self.rhythm_info.tempo:.1f} BPM")
                print_value("Time signature", self.rhythm_info.time_signature)
                print_value("Confidence", f"{self.rhythm_info.confidence:.2f}")
                print_value("Beats detected", len(self.rhythm_info.beat_times))

            self.log_step("Rhythm Engine", "success", duration=elapsed)
            return True

        except Exception as e:
            self.log_step("Rhythm Engine", "failed", str(e))
            print_error(f"Rhythm engine failed: {e}")
            traceback.print_exc()
            return False

    async def test_drum_intelligence(self):
        """Test drum intelligence with parameter inspection."""
        try:
            from modules.drum_intelligence import DrumIntelligence

            print_info("Testing DrumIntelligence...")
            start = time.time()

            if self.stems is None or 'drums' not in self.stems:
                print_warning("No drum stem available, using full audio")
                drum_audio = self.audio
            else:
                drum_audio = self.stems['drums']

            engine = DrumIntelligence()

            # Try different call signatures
            try:
                result = engine.process(
                    audio=drum_audio,
                    sr=self.sr,
                    state=None,
                    use_validation=True
                )
                self.drum_hits = result if isinstance(result, tuple) else result
                print_success("DrumIntelligence works with state=None")

            except TypeError as e:
                print_warning(f"Parameter issue: {e}")
                try:
                    result = engine.process(
                        audio=drum_audio,
                        sr=self.sr
                    )
                    self.drum_hits = result if isinstance(result, tuple) else result
                    print_success("DrumIntelligence works without state")
                except Exception as e2:
                    raise e2

            elapsed = time.time() - start
            if self.drum_hits:
                hit_count = len(self.drum_hits) if isinstance(self.drum_hits, list) else 0
                print_value("Drum hits detected", hit_count)

            self.log_step("Drum Intelligence", "success", duration=elapsed)
            return True

        except Exception as e:
            self.log_step("Drum Intelligence", "failed", str(e))
            print_error(f"Drum intelligence failed: {e}")
            traceback.print_exc()
            return False

    async def test_pitch_intelligence(self):
        """Test pitch intelligence with parameter inspection."""
        try:
            from modules.pitch_intelligence import PitchIntelligence

            print_info("Testing PitchIntelligence...")
            start = time.time()

            # Get piano stem
            if self.stems and 'piano' in self.stems:
                piano_audio = self.stems['piano']
            else:
                piano_audio = self.audio
                print_warning("No piano stem, using full audio")

            engine = PitchIntelligence()

            # Create a temp file for Basic Pitch
            import tempfile
            import soundfile as sf

            temp_path = None
            if piano_audio is not None and len(piano_audio) > 0:
                temp_path = Path(tempfile.gettempdir()) / "debug_piano.wav"
                sf.write(str(temp_path), piano_audio, self.sr)
                print_info(f"Created temp file: {temp_path}")

            # Test process_piano with different signatures
            tempo = self.rhythm_info.tempo if self.rhythm_info else 120.0

            try:
                # Try with state parameter
                result = await engine.process_piano(
                    audio=piano_audio,
                    sr=self.sr,
                    audio_path=temp_path,
                    tempo=tempo,
                    state=None
                )
                self.notes = result
                print_success("PitchIntelligence works with state=None")

            except TypeError as e:
                print_warning(f"State parameter issue: {e}")
                # Try without state
                try:
                    result = await engine.process_piano(
                        audio=piano_audio,
                        sr=self.sr,
                        audio_path=temp_path,
                        tempo=tempo
                    )
                    self.notes = result
                    print_success("PitchIntelligence works without state")
                except Exception as e2:
                    raise e2

            elapsed = time.time() - start

            # Analyze results
            if self.notes:
                # Check if these are placeholder notes (C major scale pattern)
                pitches = [n.pitch for n in self.notes[:20]]
                is_placeholder = all(60 <= p <= 67 for p in pitches) and len(pitches) > 10

                if is_placeholder:
                    print_error(f"⚠️ DETECTED PLACEHOLDER NOTES! Pitch detection failed!")
                    print_value("Sample pitches", pitches[:10])
                    print_warning("Real pitch detection is not running - check parameter passing")
                else:
                    print_success(f"Real notes detected: {len(self.notes)} notes")
                    print_value("Sample pitches", pitches[:10])
            else:
                print_error("No notes detected at all!")

            print_value("Total notes", len(self.notes) if self.notes else 0)
            self.log_step("Pitch Intelligence", "success" if self.notes else "warning",
                          f"{len(self.notes)} notes" if self.notes else "no notes", elapsed)

            # Cleanup
            if temp_path and temp_path.exists():
                temp_path.unlink()

            return self.notes is not None and len(self.notes) > 0

        except Exception as e:
            self.log_step("Pitch Intelligence", "failed", str(e))
            print_error(f"Pitch intelligence failed: {e}")
            traceback.print_exc()
            return False

    async def test_full_pipeline(self):
        """Run the full pipeline and capture everything."""
        print_section("3. FULL PIPELINE EXECUTION")

        try:
            from grimlock_pipeline import create_pipeline

            print_info("Creating pipeline...")

            # Create progress tracker
            progress_log = []

            def on_progress(percent: int, message: str):
                progress_log.append((percent, message))
                print(f"   [{percent:3d}%] {message}")

            # Create pipeline with debug mode
            pipeline = create_pipeline(
                debug=True,
                progress_callback=on_progress
            )

            print_info("Running pipeline.process()...")
            print_info(f"Audio: {self.audio_path}")
            print_info(f"Truncate: {self.truncate_seconds}s")
            start = time.time()

            result = await pipeline.process(
                audio_path=self.audio_path,
                truncate_seconds=self.truncate_seconds
            )

            elapsed = time.time() - start

            print_section("4. RESULTS ANALYSIS")

            # Analyze result
            print_value("Task ID", result.task_id)
            print_value("Tempo", f"{result.tempo:.1f} BPM")
            print_value("Time signature", result.time_signature)
            print_value("Key", result.key)
            print_value("Confidence score", f"{result.confidence_score:.2f}")
            print_value("Notes detected", len(result.notes))
            print_value("Drum hits", len(result.drum_hits))
            print_value("Success", result.success)
            print_value("Warnings", len(result.warnings))

            # Check for placeholder notes again
            if result.notes:
                first_pitches = [n.pitch for n in result.notes[:20]]
                all_pitches = [n.pitch for n in result.notes[:100]]
                unique_pitches = set(all_pitches)

                print_info("Note analysis:")
                print_value("First 20 pitches", first_pitches)
                print_value("Unique pitches (first 100 notes)", sorted(unique_pitches))

                # Detect placeholder pattern
                if len(unique_pitches) <= 8 and all(60 <= p <= 67 for p in unique_pitches):
                    print_error("⚠️ WARNING: Notes appear to be PLACEHOLDER NOTES!")
                    print_error("   Real pitch detection is NOT working!")
                else:
                    print_success("Real pitch detection appears to be working!")

            # Show progress steps
            print_info("Progress steps:")
            for percent, msg in progress_log[:20]:
                print(f"   [{percent}%] {msg}")

            self.result = result
            self.log_step("Full Pipeline", "success",
                          f"{len(result.notes)} notes, {len(result.drum_hits)} drums", elapsed)

            return result

        except Exception as e:
            self.log_step("Full Pipeline", "failed", str(e))
            print_error(f"Pipeline failed: {e}")
            traceback.print_exc()
            return None

    def print_summary(self):
        """Print final diagnostic summary."""
        print_section("5. DIAGNOSTIC SUMMARY")

        successful_steps = [s for s in self.steps if s['status'] == 'success']
        failed_steps = [s for s in self.steps if s['status'] == 'failed']
        warning_steps = [s for s in self.steps if s['status'] == 'warning']

        print_value("Total steps", len(self.steps))
        print_value("Successful", len(successful_steps))
        print_value("Failed", len(failed_steps))
        print_value("Warnings", len(warning_steps))

        if failed_steps:
            print_error("\nFailed steps:")
            for step in failed_steps:
                print(f"   - {step['name']}: {step['details']}")

        # Root cause analysis
        if self.notes and len(self.notes) > 0:
            first_pitches = [n.pitch for n in self.notes[:20]]
            if all(60 <= p <= 67 for p in first_pitches):
                print_error("\n🔴 ROOT CAUSE IDENTIFIED:")
                print_error("   Pitch detection is producing PLACEHOLDER NOTES, not real transcription!")
                print_error("   This means PitchIntelligence.process_piano() is failing silently.")
                print_error("   Most likely: Missing 'state' parameter in pitch_intelligence.py")
                print_error("")
                print_info("FIX: Add 'state=None' parameter to:")
                print_info("   - PitchIntelligence.process()")
                print_info("   - PitchIntelligence.process_piano()")
                print_info("   - PitchIntelligence.process_bass()")
                print_info("   - PitchIntelligence.process_melody()")

                # Show the exact fix
                print_info("\nEXAMPLE FIX for pitch_intelligence.py:")
                print("```python")
                print(
                    "async def process_piano(self, audio, sr, audio_path=None, tempo=120.0, audio_16k=None, state=None):")
                print("    notes = await self.process(audio, sr, audio_path, audio_16k, state=state)")
                print("    for note in notes:")
                print("        note.instrument = InstrumentType.PIANO")
                print("    return notes")
                print("```")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grimlock Pipeline Debugger")
    parser.add_argument("--audio", "-a", type=str,
                        default=str(DEFAULT_AUDIO),
                        help=f"Path to audio file (default: {DEFAULT_AUDIO})")
    parser.add_argument("--truncate", "-t", type=int, default=30,
                        help="Truncate to N seconds (default: 30)")
    parser.add_argument("--skip-imports", action="store_true",
                        help="Skip import tests")
    parser.add_argument("--skip-signatures", action="store_true",
                        help="Skip signature inspection")
    parser.add_argument("--list-audio", action="store_true",
                        help="List available audio files in Downloads")

    args = parser.parse_args()

    # List available audio files if requested
    if args.list_audio:
        print_section("📁 AVAILABLE AUDIO FILES")
        downloads = Path(r"C:\Users\kiwi2\Downloads")
        if downloads.exists():
            audio_files = list(downloads.glob("*.mp3")) + list(downloads.glob("*.wav")) + list(downloads.glob("*.m4a"))
            for f in audio_files[:20]:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   {f.name} ({size_mb:.1f} MB)")
        return

    # Check audio file
    audio_path = Path(args.audio)
    checked_path = check_audio_file(audio_path)

    if checked_path is None:
        print_error("Cannot proceed without valid audio file")
        print_info(f"Usage: python grimlock_debug.py --audio \"path/to/audio.mp3\"")
        print_info(f"Or place Hopeful.mp3 in: {DEFAULT_AUDIO}")
        return

    audio_path = checked_path

    print_section("🔍 GRIMLOCK PIPELINE DEBUGGER")
    print_value("Audio file", audio_path)
    print_value("File size", f"{audio_path.stat().st_size / 1024 / 1024:.2f} MB")
    print_value("Truncate", f"{args.truncate}s" if args.truncate > 0 else "full")

    # Import tests
    if not args.skip_imports:
        test_imports()

    # Signature inspection
    if not args.skip_signatures:
        inspect_method_signatures()

    # Create debugger
    debugger = PipelineDebugger(audio_path, args.truncate)

    # Run tests
    print_section("PIPELINE STEP TESTS")

    # Test audio loading
    if not await debugger.test_audio_loading():
        print_error("Audio loading failed - cannot continue")
        return

    # Test rhythm engine
    await debugger.test_rhythm_engine()

    # Test drum intelligence
    await debugger.test_drum_intelligence()

    # Test pitch intelligence (critical!)
    pitch_working = await debugger.test_pitch_intelligence()

    # Full pipeline test
    await debugger.test_full_pipeline()

    # Print summary
    debugger.print_summary()

    # Final advice
    if not pitch_working:
        print_section("🔧 RECOMMENDED FIXES")
        print("1. Add 'state=None' parameter to all PitchIntelligence methods:")
        print("   - process(self, ..., state=None)")
        print("   - process_piano(self, ..., state=None)")
        print("   - process_bass(self, ..., state=None)")
        print("   - process_melody(self, ..., state=None)")
        print()
        print("2. In pitch_intelligence.py, ensure the process() method passes state")
        print("   to individual detectors if they need it (optional)")
        print()
        print("3. Restart the server and re-run this debugger")
        print("   You should see REAL notes instead of placeholder (60-67) notes")
    else:
        print_section("✅ PIPELINE APPEARS HEALTHY")
        print("If you're still having issues, check:")
        print("   - Are you getting real note pitches (not just 60-67)?")
        print("   - Is tempo detection working?")
        print("   - Are drums being detected?")


if __name__ == "__main__":
    import numpy as np

    asyncio.run(main())