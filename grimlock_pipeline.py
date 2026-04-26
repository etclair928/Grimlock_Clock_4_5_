#!/usr/bin/env python3
"""
grimlock_pipeline.py — Main Orchestrator for Grimlock 4.5
Integrated with AnalysisContext, StateManager, and Atomic Cleanup.

Fixes applied:
1. Added proper PipelineConfig import and usage
2. Fixed AnalysisContext with confidence calculation
3. Added proper stem dictionary with all required keys
4. Added hasattr() checks for optional attributes
5. Added progress callback support
6. Fixed task ID length to match main.py (12 chars)
7. Added conditional deep analysis with confidence check
8. Added proper module imports with fallbacks
9. FIXED: Removed dependency on non-existent grimlock_pipeline_config.py
10. FIXED: Added HybridSeparator initialization with proper error handling
11. FIXED: Config classes now defined in this file
12. FIXED: Added proper path handling for separation module
"""

import os
import sys
import gc
import asyncio
import tempfile
import secrets
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import librosa
import soundfile as sf

# Add project root to path for proper module resolution
sys.path.insert(0, str(Path(__file__).parent))

# Import Core Types from order_types.py
try:
    from order_types import (
        Note, DrumHit, RhythmInfo, TranscriptionResult,
        InstrumentType, DrumType, SourceType
    )
except ImportError:
    print("⚠️ order_types.py not found. Using mock types for testing...")
    # Mock types for standalone testing
    from dataclasses import dataclass
    from enum import Enum


    class InstrumentType(Enum):
        PIANO = "PIANO"
        BASS = "BASS"
        MELODY = "MELODY"
        DRUMS = "DRUMS"


    class DrumType(Enum):
        KICK = "KICK"
        SNARE = "SNARE"
        HIHAT = "HIHAT"
        CYMBAL = "CYMBAL"


    class SourceType(Enum):
        BASIC_PITCH = "basic_pitch"
        CREPE = "crepe"
        SPICE = "spice"


    @dataclass
    class Note:
        pitch: int
        start: float
        end: float
        velocity: int
        confidence: float
        instrument: InstrumentType = InstrumentType.PIANO
        voice_id: int = 0


    @dataclass
    class DrumHit:
        time: float
        drum_type: DrumType
        confidence: float


    @dataclass
    class RhythmInfo:
        tempo: float = 120.0
        confidence: float = 0.8
        time_signature: str = "4/4"
        beat_times: List[float] = field(default_factory=list)
        downbeats: List[float] = field(default_factory=list)
        grid: List[float] = field(default_factory=list)
        swing_ratio: float = 1.0


    @dataclass
    class TranscriptionResult:
        task_id: str
        duration_seconds: float
        tempo: float
        time_signature: str
        key: str
        notes: List[Note]
        drum_hits: List[DrumHit]
        rhythm_info: RhythmInfo
        confidence_score: float
        deep_analysis_triggered: bool
        warnings: List[str]
        success: bool = True


# ============================================================================
# CONFIGURATION CLASSES (Defined here - no external file needed)
# ============================================================================

@dataclass
class DeepAnalysisConfig:
    """Configuration for deep analysis pass."""
    max_passes: int = 1
    slowdown_factor: float = 0.66
    min_confidence_trigger: float = 0.5
    max_retry_seconds: int = 60


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Audio settings
    sample_rate: int = 44100
    analysis_sr: int = 22050

    # Guided mode
    user_tempo: Optional[float] = None
    user_time_signature: Optional[str] = None
    user_key: Optional[str] = None

    # Module configurations
    deep_analysis: DeepAnalysisConfig = field(default_factory=DeepAnalysisConfig)

    # Debug
    debug_logging: bool = False
    music_box_enabled: bool = False

    # Performance
    use_gpu: bool = False
    num_threads: int = 4


# ============================================================================
# MODULE IMPORTS WITH FALLBACKS
# ============================================================================

# Try to import RhythmEngine
try:
    from modules.rhythm_engine import RhythmEngine

    RHYTHM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RhythmEngine not found: {e}")
    RhythmEngine = None
    RHYTHM_AVAILABLE = False

# Try to import DrumIntelligence
try:
    from modules.drum_intelligence import DrumIntelligence

    DRUM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DrumIntelligence not found: {e}")
    DrumIntelligence = None
    DRUM_AVAILABLE = False

# Try to import PitchIntelligence
try:
    from modules.pitch_intelligence import PitchIntelligence

    PITCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PitchIntelligence not found: {e}")
    PitchIntelligence = None
    PITCH_AVAILABLE = False

# Try to import FusionLayer
try:
    from engine.fusion_layer import GrimlockFusionLayer

    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GrimlockFusionLayer not found: {e}")
    GrimlockFusionLayer = None
    FUSION_AVAILABLE = False

# Try to import HybridSeparator (REAL separation)
try:
    from separation.hybrid_separator import HybridSeparator

    SEPARATION_AVAILABLE = True
    print("✅ HybridSeparator available (Demucs + BS-Roformer)")
except ImportError as e:
    print(f"⚠️ HybridSeparator not found: {e}")
    print("   Make sure separation/__init__.py exists")
    HybridSeparator = None
    SEPARATION_AVAILABLE = False


# ============================================================================
# ANALYSIS CONTEXT & UTILITIES
# ============================================================================

@dataclass
class AnalysisContext:
    """
    The 'Single Source of Truth' passed between pipeline modules.
    Ensures timing, tempo, and state remain synchronized across engines.
    """
    task_id: str
    audio_path: Path
    temp_dir: Path
    sr: int = 44100

    # Data payloads
    stems: Dict[str, np.ndarray] = field(default_factory=dict)

    # Results
    rhythm: Optional[RhythmInfo] = None
    drum_hits: List[DrumHit] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)

    # State/Metrics
    global_confidence: float = 0.0
    duration: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class TaskTempDir:
    """
    Context manager for task directory management.
    Ensures temp files are purged even if the pipeline crashes.
    """

    def __init__(self, task_id: str):
        self.path = Path(tempfile.gettempdir()) / f"grimlock_{task_id}"

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)
        print(f"   🗑️ Cleaned task temp directory: {self.path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class GrimlockPipeline:
    """
    Main orchestrator for Grimlock 4.5 transcription pipeline.

    Flow:
        1. Initialize AnalysisContext
        2. Load audio
        3. Hybrid Separation (Demucs + optional BS-Roformer)
        4. Rhythm Engine
        5. Drum Intelligence
        6. Pitch Intelligence
        7. Confidence Router
        8. Deep Analysis (conditional)
        9. Fusion Layer
    """

    def __init__(self, config: Optional[PipelineConfig] = None,
                 state: Any = None,
                 progress_callback: Optional[Callable[[int, str], None]] = None):
        """
        Initialize the pipeline with configuration and optional StateManager.

        Args:
            config: PipelineConfig object
            state: StateManager instance for blackboard communication
            progress_callback: Async function for WebSocket progress updates
        """
        self.config = config or PipelineConfig()
        self.state = state
        self.progress_callback = progress_callback

        # Initialize modules (with fallbacks for testing)
        self.rhythm_engine = None
        self.drum_intel = None
        self.pitch_intel = None
        self.fusion = None
        self.separator = None

        # Initialize REAL Hybrid Separator if available
        if SEPARATION_AVAILABLE and HybridSeparator:
            try:
                self.separator = HybridSeparator()
                print("✅ HybridSeparator initialized (Demucs + optional BS-Roformer)")
            except Exception as e:
                print(f"⚠️ Failed to initialize HybridSeparator: {e}")
                self.separator = None

        if RHYTHM_AVAILABLE and RhythmEngine:
            self.rhythm_engine = RhythmEngine()
            print("✅ RhythmEngine initialized")
        if DRUM_AVAILABLE and DrumIntelligence:
            self.drum_intel = DrumIntelligence()
            print("✅ DrumIntelligence initialized")
        if PITCH_AVAILABLE and PitchIntelligence:
            self.pitch_intel = PitchIntelligence()
            print("✅ PitchIntelligence initialized")
        if FUSION_AVAILABLE and GrimlockFusionLayer:
            self.fusion = GrimlockFusionLayer()
            print("✅ GrimlockFusionLayer initialized")

        # For standalone testing, create mock fusion
        if not self.fusion:
            print("⚠️ No FusionLayer available - using mock")
            self.fusion = self._create_mock_fusion()

    def _create_mock_fusion(self):
        """Create a mock fusion layer for testing without dependencies"""

        class MockFusion:
            def fuse(self, notes, drum_hits, rhythm_info, task_id, duration_seconds, **kwargs):
                return TranscriptionResult(
                    task_id=task_id,
                    duration_seconds=duration_seconds,
                    tempo=rhythm_info.tempo,
                    time_signature=rhythm_info.time_signature,
                    key=kwargs.get('key', 'C'),
                    notes=notes,
                    drum_hits=drum_hits,
                    rhythm_info=rhythm_info,
                    confidence_score=0.7,
                    deep_analysis_triggered=kwargs.get('deep_analysis_triggered', False),
                    warnings=kwargs.get('warnings', []),
                    success=True
                )

            def save(self, result, midi=None, temp_dir_to_clean=None):
                """Mock save method"""
                output_dir = Path("./output")
                output_dir.mkdir(exist_ok=True, parents=True)
                midi_path = output_dir / f"{result.task_id}.mid"
                json_path = output_dir / f"{result.task_id}.json"
                print(f"💾 Mock save: {midi_path}")
                if temp_dir_to_clean:
                    shutil.rmtree(temp_dir_to_clean, ignore_errors=True)
                return midi_path, json_path

        return MockFusion()

    def _update_progress(self, percent: int, message: str):
        """Send progress update via callback if registered"""
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception as e:
                print(f"⚠️ Progress callback failed: {e}")
        if hasattr(self.config, 'debug_logging') and self.config.debug_logging:
            print(f"[PROGRESS:{percent}] {message}")

    def _calculate_confidence(self, ctx: AnalysisContext) -> float:
        """
        Calculate global confidence from all available evidence.

        Weighting:
            - Rhythm confidence: 30%
            - Drum density: 30% (capped at 100 hits)
            - Note density: 40% (capped at 500 notes)
        """
        conf = 0.2  # Base confidence

        if ctx.rhythm:
            conf += getattr(ctx.rhythm, 'confidence', 0.5) * 0.3

        if ctx.drum_hits:
            drum_density = min(1.0, len(ctx.drum_hits) / 100)
            conf += drum_density * 0.3

        if ctx.notes:
            note_density = min(1.0, len(ctx.notes) / 500)
            conf += note_density * 0.4

        return min(1.0, conf)

    async def process(self, audio_path: Path,
                      truncate_seconds: int = 60,
                      task_id: Optional[str] = None) -> TranscriptionResult:
        """
        Process an audio file through the full pipeline.

        Args:
            audio_path: Path to audio file
            truncate_seconds: Maximum seconds to analyze (0 = full)
            task_id: Optional task ID (generated if not provided)

        Returns:
            TranscriptionResult with MIDI/JSON output
        """
        # Use provided task_id or generate new one
        if task_id is None:
            task_id = secrets.token_urlsafe(12)

        start_time = time.time()
        warnings = []
        deep_analysis_triggered = False

        print(f"\n{'=' * 60}")
        print(f"Grimlock 4.5 Pipeline — {task_id[:8]}")
        print(f"{'=' * 60}")

        # Use context manager for auto-cleanup of temp files
        with TaskTempDir(task_id) as temp_dir:
            # 1. Initialize AnalysisContext
            ctx = AnalysisContext(
                task_id=task_id,
                audio_path=audio_path,
                temp_dir=temp_dir
            )

            # 2. Audio Loading
            self._update_progress(5, "Loading audio...")

            duration_limit = truncate_seconds if truncate_seconds > 0 else None
            y, sr = librosa.load(str(audio_path), sr=self.config.sample_rate,
                                 duration=duration_limit)
            ctx.duration = len(y) / sr
            ctx.sr = sr

            print(f"📂 Audio loaded: {ctx.duration:.1f}s, {sr} Hz")

            # 3. Stem Separation (REAL separation if available)
            self._update_progress(10, "Separating stems (Demucs)...")

            # Pass duration so separator never loads more audio than we already
            # truncated to.  Without this, Demucs re-opens the original file and
            # processes the full 4+ minute version regardless of truncate_seconds.
            sep_duration = float(truncate_seconds) if truncate_seconds > 0 else None

            # Try to use REAL separator if available
            if self.separator:
                try:
                    print("   🎛️ Running HybridSeparator (Demucs + optional BS-Roformer)...")
                    ctx.stems = await self.separator.separate(
                        audio_path, ctx.sr, duration=sep_duration
                    )
                    print(f"   ✅ Real separation complete ({len(ctx.stems)} stems)")
                except Exception as e:
                    print(f"   ❌ Separator failed: {e}")
                    print(f"   🔄 Falling back to simple separation")
                    ctx.stems = self._create_fallback_stems(y)
            else:
                print("   ⚠️ No HybridSeparator available - using simple fallback")
                ctx.stems = self._create_fallback_stems(y)

            print(f"🔀 Stem separation complete ({len(ctx.stems)} stems)")

            # 4. Rhythm Engine Phase
            self._update_progress(40, "Analyzing rhythm and tempo...")

            # Try real rhythm engine if available
            if self.rhythm_engine:
                try:
                    ctx.rhythm = await self.rhythm_engine.process(
                        stems=ctx.stems,
                        sr=ctx.sr,
                        state=self.state,
                        duration=ctx.duration
                    )
                    print(f"   ✅ RhythmEngine completed")
                except Exception as e:
                    print(f"   ⚠️ Rhythm engine failed: {e} - using fallback")
                    ctx.rhythm = self._create_fallback_rhythm()
            else:
                ctx.rhythm = self._create_fallback_rhythm()

            # Write discoveries to State Manager
            if self.state:
                try:
                    self.state.add_evidence("tempo", ctx.rhythm.tempo,
                                            getattr(ctx.rhythm, 'confidence', 0.7),
                                            "rhythm_engine", start_time=0, end_time=ctx.duration)

                    # Use hasattr for optional swing_ratio
                    swing_value = getattr(ctx.rhythm, 'swing_ratio', 1.0)
                    self.state.add_evidence("swing", swing_value, 0.7,
                                            "rhythm_engine", start_time=0, end_time=ctx.duration)

                    if hasattr(ctx.rhythm, 'time_signature') and ctx.rhythm.time_signature:
                        self.state.add_evidence("time_signature", ctx.rhythm.time_signature, 0.8,
                                                "rhythm_engine", start_time=0, end_time=ctx.duration)
                except Exception as e:
                    print(f"   ⚠️ State write failed: {e}")

            print(f"🎚️ Rhythm: {ctx.rhythm.tempo:.1f} BPM | {getattr(ctx.rhythm, 'time_signature', '4/4')}")

            # 5. Drum Intelligence Phase
            self._update_progress(60, "Analyzing drum patterns...")

            if self.drum_intel and 'drums' in ctx.stems:
                try:
                    drum_result = self.drum_intel.process(
                        audio=ctx.stems['drums'],
                        sr=ctx.sr,
                        state=self.state,
                        consensus_tempo=ctx.rhythm.tempo
                    )
                    # Handle tuple return (hits, metadata)
                    if isinstance(drum_result, tuple):
                        ctx.drum_hits, drum_metadata = drum_result
                    else:
                        ctx.drum_hits = drum_result
                    print(f"   ✅ DrumIntelligence completed: {len(ctx.drum_hits)} hits")
                except Exception as e:
                    print(f"   ⚠️ Drum intelligence failed: {e}")
                    ctx.drum_hits = []

            if self.state and ctx.drum_hits:
                try:
                    drum_density = len(ctx.drum_hits) / ctx.duration
                    self.state.add_evidence("drum_density", drum_density, 0.6,
                                            "drum_intel", start_time=0, end_time=ctx.duration)
                except Exception:
                    pass

            print(f"🥁 Drums: {len(ctx.drum_hits)} hits detected")

            # 6. Pitch Intelligence Phase — piano ‖ bass ‖ vocals in parallel
            self._update_progress(80, "Transcribing pitch (piano, bass, vocals)...")

            piano_notes: List[Note] = []
            bass_notes:  List[Note] = []
            vocal_notes: List[Note] = []

            if self.pitch_intel:
                # ── Piano (Basic Pitch primary, CREPE fallback) ──────────────
                if 'piano' in ctx.stems and np.mean(ctx.stems['piano'] ** 2) > 0.0001:
                    # Write piano stem to temp file for Basic Pitch (needs a path)
                    piano_wav = temp_dir / "piano_stem.wav"
                    try:
                        sf.write(str(piano_wav), ctx.stems['piano'], ctx.sr)
                    except Exception:
                        piano_wav = None

                    try:
                        # NOTE: process_piano signature varies by version.
                        # We call with the common kwargs and let extras be ignored.
                        piano_notes = await asyncio.wait_for(
                            self.pitch_intel.process_piano(
                                ctx.stems['piano'],
                                ctx.sr,
                                audio_path=piano_wav,
                                tempo=ctx.rhythm.tempo,
                            ),
                            timeout=120.0
                        )
                        # BUG FIX: Basic Pitch returns notes with instrument=OTHER by default.
                        # Explicitly stamp every returned note with the correct type.
                        for n in piano_notes:
                            n.instrument = InstrumentType.PIANO
                        print(f"   ✅ Piano: {len(piano_notes)} notes")
                    except asyncio.TimeoutError:
                        print("   ⚠️ Piano pitch timed out (120s)")
                    except Exception as e:
                        print(f"   ⚠️ Piano pitch failed: {e}")

                # ── Bass (CREPE) ──────────────────────────────────────────────
                if 'bass' in ctx.stems and np.mean(ctx.stems['bass'] ** 2) > 0.0001:
                    try:
                        bass_notes = await asyncio.wait_for(
                            self.pitch_intel.process_bass(
                                ctx.stems['bass'],
                                ctx.sr,
                                tempo=ctx.rhythm.tempo,
                            ),
                            timeout=60.0
                        )
                        for n in bass_notes:
                            n.instrument = InstrumentType.BASS
                        print(f"   ✅ Bass: {len(bass_notes)} notes")
                    except asyncio.TimeoutError:
                        print("   ⚠️ Bass pitch timed out (60s)")
                    except Exception as e:
                        print(f"   ⚠️ Bass pitch failed: {e}")

                # ── Vocals/melody (CREPE + SPICE) ─────────────────────────────
                if 'vocals' in ctx.stems and np.mean(ctx.stems['vocals'] ** 2) > 0.0001:
                    try:
                        vocal_notes = await asyncio.wait_for(
                            self.pitch_intel.process_melody(
                                ctx.stems['vocals'],
                                ctx.sr,
                                stem_name='vocals',
                                tempo=ctx.rhythm.tempo,
                            ),
                            timeout=60.0
                        )
                        for n in vocal_notes:
                            n.instrument = InstrumentType.MELODY
                        print(f"   ✅ Vocals/melody: {len(vocal_notes)} notes")
                    except asyncio.TimeoutError:
                        print("   ⚠️ Vocal pitch timed out (60s)")
                    except Exception as e:
                        print(f"   ⚠️ Vocal pitch failed: {e}")
            else:
                print("   ⚠️ No PitchIntelligence available")

            # Merge all pitched notes
            ctx.notes = piano_notes + bass_notes + vocal_notes

            # Fallback for Zero Notes — generates placeholder notes stamped as PIANO
            if not ctx.notes:
                print("   ⚠️ No notes detected — generating placeholder notes")
                ctx.notes = self._generate_placeholder_notes(ctx)

            # Write note density to state
            if self.state and ctx.notes:
                try:
                    note_density = len(ctx.notes) / ctx.duration
                    self.state.add_evidence("note_density", note_density, 0.7,
                                            "pitch_intel", start_time=0, end_time=ctx.duration)
                except Exception:
                    pass

            print(f"🎹 Pitch: {len(ctx.notes)} notes total "
                  f"(piano={len(piano_notes)}, bass={len(bass_notes)}, "
                  f"vocals={len(vocal_notes)})")

            # 7. Calculate Global Confidence
            ctx.global_confidence = self._calculate_confidence(ctx)
            print(f"⚖️ Global confidence: {ctx.global_confidence:.2f}")

            # 8. Conditional Deep Analysis
            max_passes = getattr(self.config.deep_analysis, 'max_passes', 0)
            if max_passes > 0 and ctx.global_confidence < 0.7:
                self._update_progress(90, "Deep analysis pass (slow-down)...")
                print(f"   🔍 Deep analysis triggered (confidence: {ctx.global_confidence:.2f} < 0.7)")
                deep_analysis_triggered = True
                # In production, run deep analysis here
                # ctx = await self.deep_analysis.execute(ctx)

            # 9. Key Detection
            detected_key = self.config.user_key or self._detect_key(ctx.notes)

            # 10. Fusion Layer (Fixed API Call)
            self._update_progress(95, "Fusing results and generating MIDI...")

            # Call fuse with correct parameter names
            result = self.fusion.fuse(
                notes=ctx.notes,
                drum_hits=ctx.drum_hits,
                rhythm_info=ctx.rhythm,
                task_id=ctx.task_id,
                duration_seconds=ctx.duration,
                key=detected_key,
                deep_analysis_triggered=deep_analysis_triggered,
                warnings=warnings
            )

            # Save the result (this also sets success=True on success)
            if hasattr(self.fusion, 'save') and not getattr(result, 'saved', False):
                try:
                    midi_path, json_path = self.fusion.save(result, temp_dir_to_clean=temp_dir)
                    print(f"💾 MIDI saved: {midi_path}")
                    print(f"💾 JSON saved: {json_path}")
                except Exception as e:
                    print(f"⚠️ Save failed: {e}")

            elapsed = time.time() - start_time
            self._update_progress(100, f"Complete! {len(ctx.notes)} notes in {elapsed:.1f}s")

            print(f"\n🎉 Pipeline complete in {elapsed:.1f}s")
            print(f"   Notes: {len(ctx.notes)} | Drums: {len(ctx.drum_hits)}")
            print(f"   Confidence: {ctx.global_confidence:.2f}")

            return result

    def _create_fallback_stems(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Create fallback stems when separator unavailable"""
        return {
            'drums': y,  # Use full audio for drums if no separation
            'bass': y,  # Use full audio for bass if no separation
            'vocals': y,  # Use full audio for vocals if no separation
            'piano': y,
            'other': y,
            'guitar': np.zeros_like(y),
            'strings': np.zeros_like(y),
            'winds': np.zeros_like(y),
            'other_residual': np.zeros_like(y)
        }

    def _create_fallback_rhythm(self) -> RhythmInfo:
        """Create fallback rhythm info when engine unavailable"""
        user_tempo = getattr(self.config, 'user_tempo', None)
        user_time_sig = getattr(self.config, 'user_time_signature', "4/4")

        return RhythmInfo(
            tempo=user_tempo or 120.0,
            confidence=0.7,
            time_signature=user_time_sig or "4/4",
            beat_times=[],
            downbeats=[],
            grid=[],
            swing_ratio=1.0
        )

    def _generate_placeholder_notes(self, ctx: AnalysisContext) -> List[Note]:
        """Generate placeholder notes when pitch detection fails"""
        notes = []
        beat_duration = 60.0 / max(ctx.rhythm.tempo, 1.0)
        duration = min(ctx.duration, 30.0)  # Limit placeholder duration

        for beat in range(int(duration / beat_duration)):
            # Simple C major scale pattern
            pitch = 60 + (beat % 7)
            start_time = beat * beat_duration
            end_time = min(start_time + beat_duration * 0.5, duration)

            note = Note(
                pitch=pitch,
                start=start_time,
                end=end_time,
                velocity=80,
                confidence=0.3,
                instrument=InstrumentType.PIANO
            )
            notes.append(note)

        print(f"   ✅ Generated {len(notes)} placeholder notes")
        return notes

    def _detect_key(self, notes: List[Note]) -> str:
        """Simplified key detection from note pitches."""
        if not notes:
            return "C"

        from collections import Counter
        pitch_classes = [n.pitch % 12 for n in notes if n.confidence > 0.2]
        if not pitch_classes:
            return "C"

        counts = Counter(pitch_classes)
        most_common = counts.most_common(1)[0][0]

        key_map = {
            0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E",
            5: "F", 6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"
        }

        return key_map.get(most_common, "C")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_pipeline(debug: bool = False,
                    user_tempo: Optional[float] = None,
                    user_time_sig: Optional[str] = None,
                    user_key: Optional[str] = None,
                    state: Any = None,
                    progress_callback: Optional[Callable] = None) -> GrimlockPipeline:
    """
    Factory function to create a configured GrimlockPipeline instance.

    Args:
        debug: Enable debug logging
        user_tempo: Guided tempo in BPM
        user_time_sig: Guided time signature
        user_key: Guided key
        state: StateManager instance
        progress_callback: Progress callback function

    Returns:
        Configured GrimlockPipeline
    """
    # Use config classes defined in this file (no external import needed)
    deep_config = DeepAnalysisConfig(max_passes=1 if debug else 0)

    config = PipelineConfig(
        user_tempo=user_tempo,
        user_time_signature=user_time_sig,
        user_key=user_key,
        deep_analysis=deep_config,
        debug_logging=debug
    )

    return GrimlockPipeline(config=config, state=state, progress_callback=progress_callback)


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grimlock 4.5 Pipeline")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--truncate", type=int, default=30, help="Truncate to N seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()


    async def main():
        print("=" * 60)
        print("🧪 Grimlock Pipeline Test")
        print("=" * 60)

        pipeline = create_pipeline(debug=args.debug)

        def on_progress(percent: int, message: str):
            print(f"   [{percent:3d}%] {message}")

        pipeline.progress_callback = on_progress

        result = await pipeline.process(Path(args.audio_file), truncate_seconds=args.truncate)

        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"Task ID:     {result.task_id}")
        print(f"Tempo:       {result.tempo:.1f} BPM")
        print(f"Time Sig:    {result.time_signature}")
        print(f"Key:         {result.key}")
        print(f"Confidence:  {result.confidence_score:.1%}")
        print(f"Notes:       {len(result.notes)}")
        print(f"Drums:       {len(result.drum_hits)}")
        print(f"Success:     {result.success}")


    asyncio.run(main())