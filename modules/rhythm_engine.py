#!/usr/bin/env python3
"""
rhythm_engine.py — Tempo Consensus and Temporal Grid (FIXED with Madmom Timeout)
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import asyncio

# Phase 1 & 2 imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from order_types import RhythmInfo, Note, DrumHit
from core.audio_utils import ensure_float32, clamp_audio, validate_audio
from core.fft_helpers import spectral_centroid, zero_crossing_rate
from core.time_utils import (
    bpm_to_seconds_per_beat, seconds_per_beat_to_bpm,
    beats_to_seconds, seconds_to_beats,
    build_beat_grid, snap_to_grid, snap_to_grid_with_penalty,
    apply_swing, compute_ioi, estimate_tempo_stability
)

# Optional madmom import with fallback
try:
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    print("⚠️ madmom not available — using librosa fallback")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RhythmConfig:
    """Configuration for rhythm engine."""

    # Tempo consensus weights
    tempo_weights: Dict[str, float] = field(default_factory=lambda: {
        'madmom_drums': 0.6,
        'bass_onset': 0.2,
        'piano_onset': 0.2
    })

    # Tempo range
    min_tempo: float = 40.0
    max_tempo: float = 200.0

    # Grid settings
    grid_tolerance_ms: float = 65.0
    grid_penalty: float = 0.5
    subdivisions_per_beat: int = 3  # Triplet feel for jazz

    # Swing detection
    swing_min_notes: int = 10
    swing_ratio_threshold: float = 1.15

    # Beat tracking (madmom)
    madmom_fps: int = 100
    madmom_min_bpm: float = 40
    madmom_max_bpm: float = 250
    madmom_beats_per_bar: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])

    # Downbeat confidence
    min_downbeat_confidence: float = 0.4

    # Sample rate for madmom
    madmom_target_sr: int = 44100

    # Timeout for madmom operations (seconds)
    # Each madmom call (tempo, downbeats, beat_times) runs sequentially.
    # At 60s each that's 3 minutes of waiting before any grid is built.
    # 15s is enough for 60s of audio; librosa fallback fires quickly if it hangs.
    madmom_timeout: float = 15.0


# ============================================================================
# TEMPO CONSENSUS ENGINE (FIXED with better timeout handling)
# ============================================================================

class TempoConsensusEngine:
    """Weighted tempo estimation from multiple sources."""

    def __init__(self, config: RhythmConfig):
        self.config = config
        self._sources = {
            'madmom_drums': {'weight': 0.6, 'confidence': 0.0, 'tempo': None},
            'bass_onset': {'weight': 0.2, 'confidence': 0.0, 'tempo': None},
            'piano_onset': {'weight': 0.2, 'confidence': 0.0, 'tempo': None}
        }

    def estimate_from_onsets(self, audio: np.ndarray, sr: int) -> float:
        """Estimate tempo from onset detection (librosa fallback)."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        return self._clamp_tempo(tempo)

    def _clamp_tempo(self, tempo: float) -> float:
        """Clamp tempo to reasonable range."""
        if tempo < self.config.min_tempo:
            tempo *= 2
        elif tempo > self.config.max_tempo:
            tempo /= 2
        return float(np.clip(tempo, self.config.min_tempo, self.config.max_tempo))

    def add_bass_tempo(self, audio: np.ndarray, sr: int) -> None:
        """Add tempo estimate from bass stem."""
        if audio is None or len(audio) == 0:
            return

        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) > 5:
            ioi = np.diff(onset_times)
            tempo = 60.0 / np.median(ioi) if len(ioi) > 0 else 0
            if tempo > 0:
                tempo = self._clamp_tempo(tempo)
                confidence = min(0.8, len(onset_times) / 50)
                self._sources['bass_onset']['tempo'] = tempo
                self._sources['bass_onset']['confidence'] = confidence
                print(f"🎸 Bass onset tempo: {tempo:.1f} BPM (conf={confidence:.2f})")

    def add_piano_tempo(self, audio: np.ndarray, sr: int) -> None:
        """Add tempo estimate from piano/other stem."""
        if audio is None or len(audio) == 0:
            return

        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) > 5:
            ioi = np.diff(onset_times)
            tempo = 60.0 / np.median(ioi) if len(ioi) > 0 else 0
            if tempo > 0:
                tempo = self._clamp_tempo(tempo)
                confidence = min(0.8, len(onset_times) / 50)
                self._sources['piano_onset']['tempo'] = tempo
                self._sources['piano_onset']['confidence'] = confidence
                print(f"🎹 Piano onset tempo: {tempo:.1f} BPM (conf={confidence:.2f})")

    async def add_madmom_tempo(self, audio: np.ndarray, sr: int) -> None:
        """Add tempo estimate from madmom beat tracker (async with timeout)."""
        if not MADMOM_AVAILABLE:
            tempo = self.estimate_from_onsets(audio, sr)
            self._sources['madmom_drums']['tempo'] = tempo
            self._sources['madmom_drums']['confidence'] = 0.6
            print(f"🥁 Madmom fallback (librosa): {tempo:.1f} BPM")
            return

        try:
            # Run madmom in a thread with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_madmom_tempo, audio, sr),
                timeout=self.config.madmom_timeout
            )

            if result is not None:
                tempo, confidence = result
                self._sources['madmom_drums']['tempo'] = tempo
                self._sources['madmom_drums']['confidence'] = confidence
                print(f"🥁 Madmom drums tempo: {tempo:.1f} BPM (conf={confidence:.2f})")
            else:
                print(f"⚠️ Madmom returned no result — using librosa fallback")
                tempo = self.estimate_from_onsets(audio, sr)
                self._sources['madmom_drums']['tempo'] = tempo
                self._sources['madmom_drums']['confidence'] = 0.5

        except asyncio.TimeoutError:
            print(f"⚠️ Madmom tempo timeout after {self.config.madmom_timeout}s — using librosa fallback")
            tempo = self.estimate_from_onsets(audio, sr)
            self._sources['madmom_drums']['tempo'] = tempo
            self._sources['madmom_drums']['confidence'] = 0.5
        except Exception as e:
            print(f"⚠️ Madmom failed: {e}, using librosa fallback")
            tempo = self.estimate_from_onsets(audio, sr)
            self._sources['madmom_drums']['tempo'] = tempo
            self._sources['madmom_drums']['confidence'] = 0.5

    def _run_madmom_tempo(self, audio: np.ndarray, sr: int) -> Optional[Tuple[float, float]]:
        """Synchronous madmom tempo detection."""
        try:
            # Resample to 44.1kHz for madmom
            if sr != self.config.madmom_target_sr:
                audio_44k = librosa.resample(audio, orig_sr=sr, target_sr=self.config.madmom_target_sr)
            else:
                audio_44k = audio.astype(np.float32)

            import tempfile
            import soundfile as sf
            import os

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_44k, self.config.madmom_target_sr)
                tmp_path = tmp.name

            try:
                beat_proc = RNNBeatProcessor()
                beat_tracker = DBNBeatTrackingProcessor(
                    fps=self.config.madmom_fps,
                    min_bpm=self.config.madmom_min_bpm,
                    max_bpm=self.config.madmom_max_bpm,
                    transition_lambda=100
                )

                beat_act = beat_proc(tmp_path)
                beat_times = beat_tracker(beat_act)

                if len(beat_times) > 1:
                    tempo = 60.0 / np.median(np.diff(beat_times))
                    tempo = self._clamp_tempo(tempo)
                    confidence = min(0.95, len(beat_times) / 100)
                    return tempo, confidence
                else:
                    return None
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            print(f"⚠️ Madmom tempo error: {e}")
            return None

    def get_consensus_tempo(self, user_tempo: Optional[float] = None) -> float:
        """
        Compute weighted consensus tempo with outlier rejection.

        Outlier rejection: any source whose tempo is more than 15% away from
        the median of all sources is excluded from the weighted average.
        This prevents a single bad estimate (e.g. madmom detecting double-time)
        from dragging the consensus to an unusable value.
        """
        if user_tempo is not None and user_tempo > 0:
            print(f"🎚️ Guided mode: using user tempo {user_tempo:.1f} BPM")
            return float(user_tempo)

        valid = {k: d for k, d in self._sources.items()
                 if d['tempo'] is not None and d['tempo'] > 0}

        if not valid:
            return 120.0

        all_tempos = np.array([d['tempo'] for d in valid.values()])

        # Outlier rejection: exclude values > 15% from median
        median_t = float(np.median(all_tempos))
        inliers  = {k: d for k, d in valid.items()
                    if abs(d['tempo'] - median_t) / (median_t + 1e-6) <= 0.15}

        if not inliers:
            inliers = valid  # all are outliers — use them all rather than return nothing

        total_weight = 0.0
        weighted_sum = 0.0
        for source, data in inliers.items():
            weight        = data['weight'] * data['confidence']
            weighted_sum += data['tempo'] * weight
            total_weight += weight

        if total_weight > 0:
            tempo = weighted_sum / total_weight
            if len(inliers) < len(valid):
                rejected = [k for k in valid if k not in inliers]
                print(f"   ⚠️ Tempo outlier rejection: excluded {rejected}")
            print(f"🎯 Tempo consensus: {tempo:.1f} BPM")
            return float(tempo)

        return float(list(valid.values())[0]['tempo'])

    async def get_downbeats_and_time_sig(self, audio: np.ndarray, sr: int) -> Tuple[List[float], str]:
        """Get downbeats and detect time signature using madmom."""
        if not MADMOM_AVAILABLE:
            return [], "4/4"

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_madmom_downbeats, audio, sr),
                timeout=self.config.madmom_timeout
            )
            if result is not None:
                return result
            else:
                return [], "4/4"
        except asyncio.TimeoutError:
            print(f"⚠️ Madmom downbeat timeout after {self.config.madmom_timeout}s — using 4/4 default")
            return [], "4/4"
        except Exception as e:
            print(f"⚠️ Madmom downbeat detection failed: {e}")
            return [], "4/4"

    def _run_madmom_downbeats(self, audio: np.ndarray, sr: int) -> Optional[Tuple[List[float], str]]:
        """Synchronous madmom downbeat detection."""
        try:
            if sr != self.config.madmom_target_sr:
                audio_44k = librosa.resample(audio, orig_sr=sr, target_sr=self.config.madmom_target_sr)
            else:
                audio_44k = audio.astype(np.float32)

            import tempfile
            import soundfile as sf
            import os

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_44k, self.config.madmom_target_sr)
                tmp_path = tmp.name

            try:
                downbeat_proc = RNNDownBeatProcessor()
                downbeat_tracker = DBNDownBeatTrackingProcessor(
                    beats_per_bar=self.config.madmom_beats_per_bar,
                    fps=self.config.madmom_fps
                )

                db_act = downbeat_proc(tmp_path)
                db_results = downbeat_tracker(db_act)

                if db_results is not None and len(db_results) > 0:
                    db_arr = np.atleast_2d(db_results)
                    downbeats = db_arr[db_arr[:, 1] == 1, 0].tolist()

                    if len(downbeats) >= 2:
                        # Get beat times for spacing calculation
                        beat_proc = RNNBeatProcessor()
                        beat_tracker = DBNBeatTrackingProcessor(
                            fps=self.config.madmom_fps,
                            min_bpm=self.config.madmom_min_bpm,
                            max_bpm=self.config.madmom_max_bpm
                        )
                        beat_act = beat_proc(tmp_path)
                        beat_times = beat_tracker(beat_act)

                        beats_between = len([b for b in beat_times if downbeats[0] < b <= downbeats[1]])
                        time_sig_map = {2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/8", 7: "7/4"}
                        time_sig = time_sig_map.get(beats_between, "4/4")
                        return downbeats, time_sig
                    else:
                        return downbeats, "4/4"
                else:
                    return [], "4/4"
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            print(f"⚠️ Madmom downbeat error: {e}")
            return None


# ============================================================================
# SWING DETECTOR
# ============================================================================

class SwingDetector:
    """Detect swing ratio from note onset times."""

    def __init__(self, config: RhythmConfig):
        self.config = config

    def detect_swing_ratio(self, onset_times: List[float], tempo: float) -> float:
        """Detect swing ratio from note onsets."""
        if len(onset_times) < self.config.swing_min_notes:
            return 1.0

        beat_duration = 60.0 / tempo
        eighth_duration = beat_duration / 2

        intervals = np.diff(onset_times)
        nearby_8th = intervals[(intervals > eighth_duration * 0.7) &
                               (intervals < eighth_duration * 1.5)]

        if len(nearby_8th) < self.config.swing_min_notes:
            return 1.0

        median_interval = np.median(nearby_8th)
        long_8th = nearby_8th[nearby_8th > median_interval]
        short_8th = nearby_8th[nearby_8th <= median_interval]

        if len(long_8th) == 0 or len(short_8th) == 0:
            return 1.0

        ratio = np.mean(long_8th) / np.mean(short_8th)
        return float(np.clip(ratio, 1.0, 2.5))


# ============================================================================
# TEMPORAL GRID
# ============================================================================

class TemporalGrid:
    """Builds grid of expected hit positions based on tempo and time signature."""

    def __init__(self, config: RhythmConfig):
        self.config = config
        self.beat_times: List[float] = []
        self.downbeats: List[float] = []
        self.tempo: float = 120.0
        self.time_signature: str = "4/4"
        self.beats_per_bar: int = 4
        self.grid: List[float] = []
        self.beat_position_map: Dict[float, int] = {}
        self.bar_map: Dict[float, int] = {}

    def build(self, beat_times: List[float], downbeats: List[float],
              tempo: float, time_signature: str = "4/4",
              duration: float = 0.0) -> None:
        """
        Build the temporal grid.

        Fix: when beat_times and downbeats are both empty (madmom timed out
        AND no audio-derived beats), fall back to a mathematically generated
        grid from tempo + duration so the grid is never empty.
        """
        self.beat_times = beat_times
        self.downbeats  = downbeats
        self.tempo      = tempo
        self.time_signature = time_signature

        time_sig_parts = time_signature.split('/')
        self.beats_per_bar = int(time_sig_parts[0]) if len(time_sig_parts) == 2 else 4

        beat_duration = 60.0 / max(tempo, 1.0)
        sub_duration  = beat_duration / self.config.subdivisions_per_beat

        if downbeats and len(downbeats) >= 2:
            self.grid = []
            for i, downbeat in enumerate(downbeats[:-1]):
                next_downbeat = downbeats[i + 1]
                bar_duration  = next_downbeat - downbeat
                for beat in range(self.beats_per_bar):
                    beat_time = downbeat + (beat / self.beats_per_bar) * bar_duration
                    self.grid.append(beat_time)
                    self.beat_position_map[beat_time] = beat
                    self.bar_map[beat_time] = i
                    for sub in range(1, self.config.subdivisions_per_beat):
                        sub_time = beat_time + sub * sub_duration
                        self.grid.append(sub_time)
                        self.beat_position_map[sub_time] = beat
                        self.bar_map[sub_time] = i

        elif beat_times:
            self.grid = []
            for i, beat in enumerate(beat_times):
                self.grid.append(beat)
                self.beat_position_map[beat] = i % self.beats_per_bar
                self.bar_map[beat] = i // self.beats_per_bar
                for sub in range(1, self.config.subdivisions_per_beat):
                    sub_time = beat + sub * sub_duration
                    self.grid.append(sub_time)
                    self.beat_position_map[sub_time] = i % self.beats_per_bar
                    self.bar_map[sub_time] = i // self.beats_per_bar

        else:
            # FIX: mathematical fallback — generate grid from tempo alone.
            # This fires when madmom times out and librosa also returns nothing,
            # which previously left grid=[].  An empty grid means no note can
            # ever be snapped, so the pipeline produces 0 notes.
            self.grid = self._generate_mathematical_grid(
                tempo, time_signature, duration or 120.0)
            print(f"📐 Mathematical grid generated from tempo ({tempo:.1f} BPM): "
                  f"{len(self.grid)} points")

        self.grid = sorted(set(round(g, 4) for g in self.grid))
        print(f"📐 Grid built: {len(self.grid)} points, {time_signature}, "
              f"{self.beats_per_bar} beats/bar")

    def _generate_mathematical_grid(self, tempo: float, time_signature: str,
                                     duration: float) -> List[float]:
        """
        Generate a metronomic grid from tempo + time signature.
        Used when beat tracking produces no results.
        """
        beat_dur  = 60.0 / max(tempo, 1.0)
        sub_dur   = beat_dur / self.config.subdivisions_per_beat
        grid      = []
        t         = 0.0
        beat_idx  = 0
        while t <= duration + beat_dur:
            grid.append(round(t, 4))
            self.beat_position_map[round(t, 4)] = beat_idx % self.beats_per_bar
            self.bar_map[round(t, 4)]           = beat_idx // self.beats_per_bar
            for sub in range(1, self.config.subdivisions_per_beat):
                sub_t = round(t + sub * sub_dur, 4)
                grid.append(sub_t)
                self.beat_position_map[sub_t] = beat_idx % self.beats_per_bar
                self.bar_map[sub_t]           = beat_idx // self.beats_per_bar
            t        += beat_dur
            beat_idx += 1
        return grid

    def shift_to_drum_hits(self, hits: List, max_shift_ms: float = 30.0) -> None:
        """
        Adaptive grid: shift all grid points by the median offset between
        drum hits and their nearest grid point.

        Jazz drummers often play slightly ahead of or behind the mathematical
        grid.  This method clusters the deviations and shifts the whole grid
        so snapping works with the drummer's feel rather than against it.

        Args:
            hits:          DrumHit list (or any list with a .time attribute)
            max_shift_ms:  Maximum allowed shift (caps runaway corrections)
        """
        if not hits or not self.grid:
            return

        grid_arr = np.array(self.grid)
        offsets  = []
        for hit in hits[:200]:   # sample first 200 hits — enough to find pattern
            t       = hit.time if hasattr(hit, 'time') else hit.get('time', 0)
            nearest = float(grid_arr[np.argmin(np.abs(grid_arr - t))])
            offsets.append(t - nearest)

        if not offsets:
            return

        median_offset = float(np.median(offsets))
        max_shift     = max_shift_ms / 1000.0

        if abs(median_offset) < 0.005:    # < 5 ms — not worth shifting
            return

        # Cap the shift
        shift = np.clip(median_offset, -max_shift, max_shift)
        self.grid = [round(g + shift, 4) for g in self.grid]
        # Rebuild position maps (shifted times are new keys)
        old_pos  = self.beat_position_map
        old_bar  = self.bar_map
        self.beat_position_map = {}
        self.bar_map           = {}
        for new_t, (old_t, pos) in zip(
                self.grid, zip(sorted(old_pos), [old_pos[k] for k in sorted(old_pos)])):
            self.beat_position_map[new_t] = pos
        for new_t, (old_t, bar) in zip(
                self.grid, zip(sorted(old_bar), [old_bar[k] for k in sorted(old_bar)])):
            self.bar_map[new_t] = bar
        print(f"📐 Grid shifted {shift * 1000:+.1f} ms to align with drummer's feel")

    def apply_penalty(self, hit_time: float) -> Tuple[float, int, float]:
        """Apply confidence penalty for off-grid hits."""
        if not self.grid:
            return hit_time, -1, 1.0

        tolerance = self.config.grid_tolerance_ms / 1000.0
        grid_array = np.array(self.grid)

        idx = int(np.argmin(np.abs(grid_array - hit_time)))
        dist = abs(self.grid[idx] - hit_time)

        if dist <= tolerance:
            snapped_time = self.grid[idx]
            beat_position = self.beat_position_map.get(snapped_time, -1)
            return snapped_time, beat_position, 1.0
        else:
            return hit_time, -1, self.config.grid_penalty

    def snap_drum_hit(self, hit: DrumHit) -> DrumHit:
        """Apply grid penalty to a drum hit."""
        snapped_time, beat_pos, multiplier = self.apply_penalty(hit.time)
        hit.time = snapped_time
        hit.beat_position = beat_pos
        hit.confidence = min(1.0, hit.confidence * multiplier)
        hit.grid_deviation_ms = 0.0 if multiplier >= 1.0 else self.config.grid_tolerance_ms
        return hit

    def snap_note(self, note: Note) -> Note:
        """Apply grid penalty to a pitched note."""
        snapped_time, beat_pos, multiplier = self.apply_penalty(note.start)
        note.start = snapped_time
        note.confidence = min(1.0, note.confidence * multiplier)
        return note


# ============================================================================
# MAIN RHYTHM ENGINE (FIXED)
# ============================================================================

class RhythmEngine:
    """Main rhythm engine orchestrating tempo consensus, grid building, and swing detection."""

    def __init__(self, config: Optional[RhythmConfig] = None):
        self.config = config or RhythmConfig()
        self.tempo_engine = TempoConsensusEngine(self.config)
        self.swing_detector = SwingDetector(self.config)
        self.grid = TemporalGrid(self.config)

        self._tempo: float = 120.0
        self._time_signature: str = "4/4"
        self._beat_times: List[float] = []
        self._downbeats: List[float] = []

    async def process(self, stems: Dict[str, np.ndarray], sr: int,
                      user_tempo: Optional[float] = None,
                      user_time_sig: Optional[str] = None,
                      state=None,           # Optional[StateManager] — avoids hard import
                      duration: float = 0.0) -> RhythmInfo:
        """
        Process rhythm from all available stems.

        Args:
            stems:         Dict of stem_name → audio array
            sr:            Sample rate
            user_tempo:    Guided-mode BPM override
            user_time_sig: Guided-mode time signature override (e.g. "4/4")
            state:         Optional StateManager — when provided, writes
                           tempo + time-signature evidence to the blackboard
            duration:      Audio duration in seconds (used for grid generation
                           fallback when beat tracking returns nothing)
        """
        print("   🔍 RhythmEngine: Starting tempo consensus...")

        # Safe numpy-array-aware stem extraction.
        # Never use `stems.get('x') or stems.get('y')` with numpy arrays —
        # the `or` operator calls __bool__ which raises
        # "The truth value of an array is ambiguous".
        def _safe_stem(keys):
            """Return first non-None, non-empty stem from `keys`."""
            for k in (keys if isinstance(keys, (list, tuple)) else [keys]):
                val = stems.get(k)
                if val is not None and isinstance(val, np.ndarray) and val.size > 0:
                    return val
            return None

        drums_audio = _safe_stem('drums')
        bass_audio  = _safe_stem('bass')
        other_audio = _safe_stem(['piano', 'other'])

        if drums_audio is not None and len(drums_audio) > 0:
            print("   🔍 Getting madmom tempo...")
            await self.tempo_engine.add_madmom_tempo(drums_audio, sr)

        if bass_audio is not None and len(bass_audio) > 0:
            print("   🔍 Getting bass tempo...")
            self.tempo_engine.add_bass_tempo(bass_audio, sr)

        if other_audio is not None and len(other_audio) > 0:
            print("   🔍 Getting piano tempo...")
            self.tempo_engine.add_piano_tempo(other_audio, sr)

        self._tempo = self.tempo_engine.get_consensus_tempo(user_tempo)
        print(f"   🔍 Consensus tempo: {self._tempo:.1f} BPM")

        # Write tempo evidence to StateManager
        if state and not user_tempo:
            try:
                state.add_evidence("tempo", self._tempo, 0.75, "rhythm_engine",
                                   start_time=0.0, end_time=duration)
            except Exception:
                pass

        # Step 2: Get downbeats and time signature
        print("   🔍 Getting downbeats and time signature...")
        if user_time_sig:
            self._time_signature = user_time_sig
            print(f"   🎚️ Guided mode: using user time signature {self._time_signature}")
            downbeats  = []
            beat_times = []
        elif drums_audio is not None and len(drums_audio) > 0:
            downbeats, self._time_signature = await self.tempo_engine.get_downbeats_and_time_sig(
                drums_audio, sr
            )
            beat_times = await self._get_beat_times(drums_audio, sr)
            print(f"   🔍 Detected time signature: {self._time_signature}")
        else:
            downbeats  = []
            beat_times = []
            self._time_signature = "4/4"

        self._downbeats  = downbeats
        self._beat_times = beat_times

        # Write time signature evidence
        if state and not user_time_sig:
            try:
                ts_conf = 0.8 if downbeats else 0.4
                state.add_evidence("time_signature", self._time_signature, ts_conf,
                                   "rhythm_engine", start_time=0.0, end_time=duration)
            except Exception:
                pass

        # Step 3: Build temporal grid — now passes duration for mathematical fallback
        print("   🔍 Building grid...")
        self.grid.build(beat_times, downbeats, self._tempo, self._time_signature,
                        duration=duration)

        # Step 4: Detect swing
        print("   🔍 Detecting swing...")
        all_onsets = []
        if bass_audio is not None and len(bass_audio) > 0:
            try:
                all_onsets.extend(
                    librosa.onset.onset_detect(y=bass_audio, sr=sr, units='time').tolist())
            except Exception:
                pass
        if other_audio is not None and len(other_audio) > 0:
            try:
                all_onsets.extend(
                    librosa.onset.onset_detect(y=other_audio, sr=sr, units='time').tolist())
            except Exception:
                pass

        swing_ratio = self.swing_detector.detect_swing_ratio(all_onsets, self._tempo)
        print(f"   🔍 Swing ratio: {swing_ratio:.2f}")

        # Step 5: Build RhythmInfo
        rhythm_info = RhythmInfo(
            tempo=self._tempo,
            beat_times=self._beat_times,
            downbeats=self._downbeats,
            time_signature=self._time_signature,
            beats_per_bar=self.grid.beats_per_bar,
            confidence=self._calculate_confidence(),
            grid=self.grid.grid,
            tracker_source='rhythm_engine',
        )

        # Write swing evidence to StateManager
        if state and swing_ratio > 0:
            try:
                state.add_evidence("swing", swing_ratio, 0.6, "rhythm_engine",
                                   start_time=0.0, end_time=duration)
            except Exception:
                pass

        print("   ✅ RhythmEngine complete")
        return rhythm_info

    async def _get_beat_times(self, audio: np.ndarray, sr: int) -> List[float]:
        """Get beat times using madmom or librosa fallback."""
        if MADMOM_AVAILABLE:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._run_madmom_beat_times, audio, sr),
                    timeout=self.config.madmom_timeout
                )
            except asyncio.TimeoutError:
                print(f"⚠️ Madmom beat times timeout after {self.config.madmom_timeout}s — using librosa fallback")
            except Exception as e:
                print(f"⚠️ Madmom beat times failed: {e}")

        # Librosa fallback
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return librosa.frames_to_time(beats, sr=sr).tolist()

    def _run_madmom_beat_times(self, audio: np.ndarray, sr: int) -> List[float]:
        """Synchronous madmom beat times."""
        if sr != self.config.madmom_target_sr:
            audio_44k = librosa.resample(audio, orig_sr=sr, target_sr=self.config.madmom_target_sr)
        else:
            audio_44k = audio.astype(np.float32)

        import tempfile
        import soundfile as sf
        import os

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_44k, self.config.madmom_target_sr)
            tmp_path = tmp.name

        try:
            beat_proc = RNNBeatProcessor()
            beat_tracker = DBNBeatTrackingProcessor(
                fps=self.config.madmom_fps,
                min_bpm=self.config.madmom_min_bpm,
                max_bpm=self.config.madmom_max_bpm
            )
            beat_act = beat_proc(tmp_path)
            beat_times = beat_tracker(beat_act)
            return beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    def _calculate_confidence(self) -> float:
        """Calculate overall rhythm confidence."""
        conf = 0.7
        if self._downbeats and len(self._downbeats) > 1:
            conf = min(1.0, conf + 0.15)
        if 60 <= self._tempo <= 180:
            conf = min(1.0, conf + 0.1)
        return conf

    def snap_drum_hits(self, hits: List[DrumHit], adaptive: bool = True) -> List[DrumHit]:
        """
        Apply grid snapping to all drum hits.

        If adaptive=True (default), first shifts the grid to align with the
        drummer's natural feel before snapping individual hits.
        """
        if adaptive and hits:
            self.grid.shift_to_drum_hits(hits)
        return [self.grid.snap_drum_hit(hit) for hit in hits]

    def snap_notes(self, notes: List[Note]) -> List[Note]:
        """Apply grid snapping to all pitched notes."""
        return [self.grid.snap_note(note) for note in notes]


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Test Rhythm Engine")
    parser.add_argument("audio_file", help="Path to audio file (drums recommended)")
    args = parser.parse_args()


    async def test():
        audio, sr = librosa.load(args.audio_file, sr=22050)
        rhythm = RhythmEngine()
        stems = {'drums': audio}
        info = await rhythm.process(stems, sr)

        print(f"\n{'=' * 60}")
        print(f"Rhythm Engine Test")
        print(f"{'=' * 60}")
        print(f"Tempo: {info.tempo:.1f} BPM")
        print(f"Time signature: {info.time_signature}")
        print(f"Confidence: {info.confidence:.2f}")
        print(f"Grid points: {len(info.grid)}")


    asyncio.run(test())