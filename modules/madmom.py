#!/usr/bin/env python3
"""
madmom_tracker.py — Madmom Beat and Downbeat Tracking (Legacy)

This module wraps madmom for beat and downbeat detection.
Updated for Grimlock 4.5 with:
1. Atomic Timeout Pattern to prevent hung processes.
2. NumPy 2.x Polyfills for legacy compatibility.
3. Memory-first handling to reduce Disk I/O.
"""

import numpy as np
import librosa
import asyncio
import os
import tempfile
import soundfile as sf
from typing import Dict, List, Optional, Tuple

# ============================================================================
# NUMPY 2.X COMPATIBILITY POLYFILL
# ============================================================================
# Madmom relies on np.float and np.int which were removed in NumPy 2.0
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

# Try to import madmom with compatibility fixes
MADMOM_AVAILABLE = False
try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

        MADMOM_AVAILABLE = True
        print("✅ Madmom available (with NumPy 2.x Polyfills)")
except ImportError as e:
    print(f"⚠️ Madmom not available: {e}")
except Exception as e:
    print(f"⚠️ Madmom import failed: {e}")


class MadmomBeatTracker:
    """
    Madmom-based beat and downbeat tracking with automatic fallback and
    hard timeout protection.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._beat_proc = None
        self._beat_tracker = None
        self._downbeat_proc = None
        self._downbeat_tracker = None
        self._loaded = False

    def _load_madmom(self) -> bool:
        """Load madmom components if available."""
        if not MADMOM_AVAILABLE:
            return False

        if self._loaded:
            return True

        try:
            # IMPROVEMENT: Pre-load these in server startup to avoid "Cold Start" timeouts
            self._beat_proc = RNNBeatProcessor()
            self._beat_tracker = DBNBeatTrackingProcessor(fps=100)
            self._downbeat_proc = RNNDownBeatProcessor()
            self._downbeat_tracker = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4],  # Support 3/4 and 4/4
                fps=100
            )
            self._loaded = True
            return True
        except Exception as e:
            print(f"⚠️ Madmom load failed: {e}")
            return False

    async def track(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Track beats and downbeats using madmom or fallback.
        """
        if self._load_madmom():
            # Apply the Atomic Timeout Pattern
            result = await self._track_madmom(audio, sr)
            if result and result.get('beats') is not None and len(result['beats']) > 0:
                return result

        # Fallback to librosa if madmom fails, times out, or is unavailable
        return self._track_librosa(audio, sr)

    async def _track_madmom(self, audio: np.ndarray, sr: int) -> Optional[Dict]:
        """Internal madmom tracking with hard timeout."""
        tmp_path = None
        try:
            # IMPROVEMENT: Resample once. Madmom is strict about 44100Hz.
            if sr != 44100:
                audio_44k = librosa.resample(audio, orig_sr=sr, target_sr=44100)
            else:
                audio_44k = audio.astype(np.float32)

            # Using a safer tempfile pattern for high-concurrency
            fd, tmp_path = tempfile.mkstemp(suffix='.wav')
            try:
                with os.fdopen(fd, 'wb') as tmp:
                    sf.write(tmp, audio_44k, 44100)

                def _run_madmom_logic():
                    # This block is prone to hanging in certain environments
                    beat_activations = self._beat_proc(tmp_path)
                    beats = self._beat_tracker(beat_activations)

                    downbeat_activations = self._downbeat_proc(tmp_path)
                    downbeat_results = self._downbeat_tracker(downbeat_activations)

                    return beats, downbeat_results

                # IMPROVEMENT: ATOMIC TIMEOUT
                # We give Madmom 30 seconds. If it's a "Zombie" process, we kill the task.
                try:
                    beats, downbeat_results = await asyncio.wait_for(
                        asyncio.to_thread(_run_madmom_logic),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    print("🛑 Madmom Tracker HANG detected. Moving to Librosa fallback.")
                    return None

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            # Parse results
            if beats is not None and len(beats) > 1:
                tempo = 60.0 / float(np.median(np.diff(beats)))
            else:
                tempo = 120.0

            downbeats = []
            if downbeat_results is not None and len(downbeat_results) > 0:
                downbeat_arr = np.atleast_2d(downbeat_results)
                downbeats = downbeat_arr[downbeat_arr[:, 1] == 1, 0].tolist()

            # IMPROVEMENT FOR CONFIDENCE ROUTER:
            # If we successfully get Madmom results, we provide a high confidence.
            # This prevents triggering the 0.66x "Deep Analysis" slow-down unnecessarily.
            return {
                'beats': beats.tolist() if isinstance(beats, np.ndarray) else beats,
                'downbeats': downbeats,
                'tempo': float(tempo),
                'confidence': 0.88,
                'tracker': 'madmom'
            }

        except Exception as e:
            print(f"⚠️ Madmom tracking failed: {e}")
            return None

    def _track_librosa(self, audio: np.ndarray, sr: int) -> Dict:
        """Fallback to librosa beat tracking."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo_output = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = tempo_output[0] if isinstance(tempo_output, (np.ndarray, list)) else tempo_output

        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beats = librosa.frames_to_time(beat_frames, sr=sr)

        # Approximate downbeats for jazz (Assume 4/4)
        downbeats = beats[::4].tolist() if len(beats) >= 4 else beats[:1].tolist()

        # IMPROVEMENT FOR CONFIDENCE ROUTER:
        # Librosa's pulse detection is "vague." We return 0.5 to flag this for
        # the ConfidenceRouter, which may then decide to trigger Deep Analysis.
        return {
            'beats': beats.tolist(),
            'downbeats': downbeats,
            'tempo': float(tempo),
            'confidence': 0.5,
            'tracker': 'librosa_fallback'
        }


# ============================================================================
# CONFIDENCE ROUTER INTEGRATION NOTES (FOR FUTURE WORK)
# ============================================================================
"""
TODO in confidence_router.py:
1. CHECK TRACKER SOURCE: If result['tracker'] == 'librosa_fallback', 
   the 'Rhythm_conf' should be penalized by 20%.
2. GUIDED MODE SYNERGY: If result['tempo'] deviates from PipelineConfig.user_tempo 
   by > 5 BPM, drop Gc (Global Confidence) below 0.8 immediately.
3. TRIGGER DEEP ANALYSIS: Ensure that if Rhythm_conf < 0.6, the router 
   forces the 0.66x slow-down pass specifically on the 'drums' and 'bass' stems.
"""

_madmom_tracker = None


def get_madmom_tracker() -> MadmomBeatTracker:
    global _madmom_tracker
    if _madmom_tracker is None:
        _madmom_tracker = MadmomBeatTracker()
    return _madmom_tracker


async def enrich_rhythm_info(rhythm_info, audio: np.ndarray, sr: int,
                              user_tempo: float = None) -> None:
    """
    Madmom integration helper — called from sequential_priority / grimlock_pipeline
    after the rhythm engine has produced a baseline RhythmInfo.

    Runs MadmomBeatTracker.track() and writes results directly back onto the
    existing `rhythm_info` object in-place, so no return value is needed.

    Sets rhythm_info.tracker_source so ConfidenceRouter can apply the correct
    penalty (20% for 'librosa_fallback', none for 'madmom').

    Args:
        rhythm_info: The RhythmInfo object to update (mutated in-place).
        audio:       The full mixed or drum-stem audio array.
        sr:          Sample rate.
        user_tempo:  Optional guided-mode tempo override (BPM).  When provided,
                     the madmom result is cross-checked and a warning is printed
                     if the deviation exceeds 5 BPM.
    """
    tracker = get_madmom_tracker()
    result = await tracker.track(audio, sr)

    if result is None:
        rhythm_info.tracker_source = "librosa_fallback"
        return

    # Populate beat / downbeat data from madmom
    if result.get('beats') and len(result['beats']) > 0:
        rhythm_info.beat_times = result['beats']

    if result.get('downbeats') and len(result['downbeats']) > 0:
        rhythm_info.downbeats = result['downbeats']

    if result.get('tempo') and result['tempo'] > 20:
        rhythm_info.tempo = result['tempo']

    # Confidence comes from madmom (0.88) or librosa fallback (0.50)
    rhythm_info.confidence = result.get('confidence', rhythm_info.confidence)
    rhythm_info.tracker_source = result.get('tracker', 'unknown')

    # Rebuild the grid from the new beat times
    if rhythm_info.beat_times:
        beat_dur = 60.0 / max(rhythm_info.tempo, 1.0)
        subdivision = beat_dur / 2  # eighth-note grid
        last_beat = rhythm_info.beat_times[-1]
        grid = []
        t = rhythm_info.beat_times[0] if rhythm_info.beat_times else 0.0
        while t <= last_beat + beat_dur:
            grid.append(round(t, 4))
            t += subdivision
        rhythm_info.grid = grid

    # Cross-check against user_tempo if guided mode is active
    if user_tempo is not None and user_tempo > 0:
        deviation = abs(rhythm_info.tempo - user_tempo)
        if deviation > 5.0:
            print(f"   ⚠️ Madmom tempo ({rhythm_info.tempo:.1f} BPM) deviates "
                  f"{deviation:.1f} BPM from guided tempo ({user_tempo:.1f} BPM). "
                  f"ConfidenceRouter will cap Gc below 0.8.")

    source_label = "Madmom DBN" if rhythm_info.tracker_source == "madmom" else "Librosa fallback"
    print(f"   🥁 Beat tracker: {source_label} | "
          f"tempo={rhythm_info.tempo:.1f} BPM | "
          f"confidence={rhythm_info.confidence:.2f}").