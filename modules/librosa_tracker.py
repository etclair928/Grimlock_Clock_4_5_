#!/usr/bin/env python3
"""
librosa_tracker.py — High-Precision Independent Rhythm Intelligence

Role in Grimlock 4.5 pipeline:
  PRIMARY:   Tier-2 fallback when Madmom times out or is unavailable.
             Its output flows through enrich_rhythm_info_librosa() which
             writes back onto the shared RhythmInfo object, just like
             madmom.enrich_rhythm_info() does.

  SECONDARY: Swing ratio provider — the ONLY module that detects swing at
             the eighth-note subdivision level.  fusion_layer.py reads
             rhythm_info.swing_ratio to nudge note timings for real jazz feel.

  THIRD:     Key detection fallback when grimlock_pipeline._detect_key()
             has too few notes (now detects all 24 keys, not just 12 major).

Fixes over original:
  - Minor key templates added (jazz is often minor — Cm, Fm, Bbm, Gm...)
  - PLP time-signature code no longer uses hardcoded 120 BPM (was a bug)
  - Swing detection now measures at EIGHTH-NOTE subdivisions, not beat level
  - enrich_rhythm_info_librosa() integrates with madmom.py fallback path
  - Swing-aware grid rebuilding (long/short eighths placed correctly)
  - ConfidenceRouter-compatible tracker_source strings
  - Confidence formula penalises when < 8 beats detected
  - asyncio.to_thread used correctly throughout
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import Dict, Any, Optional, Tuple
import asyncio


# ============================================================================
# KRUMHANSL-KESSLER PROFILES  (major + minor — 24 keys total)
# ============================================================================

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

# Natural minor profile — critical for jazz (was missing in original)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F',
               'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Build all 24 templates at module load (12 major + 12 minor)
_KEY_TEMPLATES: Dict[str, np.ndarray] = {}
for _i, _note in enumerate(_NOTE_NAMES):
    _maj = np.roll(_MAJOR_PROFILE, _i)
    _min = np.roll(_MINOR_PROFILE, _i)
    _KEY_TEMPLATES[f"{_note} Major"] = _maj / _maj.sum()
    _KEY_TEMPLATES[f"{_note} Minor"] = _min / _min.sum()


# ============================================================================
# MAIN CLASS
# ============================================================================

class LibrosaBeatIntelligence:
    """
    Pure-librosa rhythm analysis with no external AI model dependencies.

    Designed as the high-quality fallback when Madmom is unavailable.
    Output format is compatible with madmom.enrich_rhythm_info() so the
    rest of the pipeline needs zero changes when this runs instead.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    async def track(self, audio: np.ndarray, sr: int,
                    segment_seconds: float = 30.0) -> Dict[str, Any]:
        """
        Analyse audio and return a dict compatible with Madmom's output.

        Returns:
            beats, downbeats, tempo, time_signature,
            swing_ratio, swing_confidence,
            detected_key, key_confidence,
            confidence, tracker
        """
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        audio = audio.astype(np.float32)

        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            self.sr = 22050
        else:
            self.sr = sr

        duration = len(audio) / self.sr
        if duration > 60:
            return await self._segmented_analysis(audio, segment_seconds)
        return await self._analyze_full(audio)

    # ── Full analysis ───────────────────────────────────────────────────────

    async def _analyze_full(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyse complete audio in one pass."""

        def _blocking():
            # Multi-band onset: low-band for stable tempo, full for time-sig
            onset_full = librosa.onset.onset_strength(
                y=audio, sr=self.sr, aggregate=np.median)
            onset_low = librosa.onset.onset_strength(
                y=audio, sr=self.sr, aggregate=np.median, fmin=40, fmax=200)

            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_low, sr=self.sr,
                start_bpm=120.0, tightness=100)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
            tempo = float(np.atleast_1d(tempo)[0])

            swing_ratio, swing_conf = self._detect_swing_ratio(
                audio, beat_times, tempo)
            time_sig, downbeats = self._detect_time_signature(
                onset_full, beat_times, tempo)
            detected_key, key_conf = self._detect_key(audio)
            confidence = self._calculate_confidence(
                tempo, beat_times, swing_ratio, key_conf)

            return {
                'beats':            beat_times.tolist(),
                'downbeats':        list(downbeats),
                'tempo':            tempo,
                'time_signature':   time_sig,
                'swing_ratio':      swing_ratio,
                'swing_confidence': swing_conf,
                'detected_key':     detected_key,
                'key_confidence':   key_conf,
                'confidence':       confidence,
                'tracker':          'librosa_fallback',
            }

        return await asyncio.to_thread(_blocking)

    # ── Segmented analysis ──────────────────────────────────────────────────

    async def _segmented_analysis(self, audio: np.ndarray,
                                  segment_seconds: float) -> Dict[str, Any]:
        """Analyse long audio in segments to avoid memory/timeout issues."""
        seg_samples = int(segment_seconds * self.sr)
        n_segments  = max(1, len(audio) // seg_samples)
        all_tempos  = []
        all_swings  = []

        for i in range(n_segments):
            start   = i * seg_samples
            end     = min(start + seg_samples, len(audio))
            segment = audio[start:end]
            if len(segment) < self.sr * 5:
                continue
            try:
                t, s = await asyncio.to_thread(self._analyze_segment, segment)
                if t > 0:
                    all_tempos.append(t)
                    all_swings.append(s)
            except Exception as e:
                print(f"   ⚠️ LibrosaTracker segment {i} failed: {e}")

        tempo       = float(np.median(all_tempos)) if all_tempos else 120.0
        swing_ratio = float(np.median(all_swings)) if all_swings else 1.0
        swing_conf  = 0.7 if len(all_swings) > 2 else 0.5

        excerpt = audio[:seg_samples * 3]
        full    = await self._analyze_full(excerpt)

        return {
            'beats':            full['beats'],
            'downbeats':        full['downbeats'],
            'tempo':            tempo,
            'time_signature':   full['time_signature'],
            'swing_ratio':      swing_ratio,
            'swing_confidence': swing_conf,
            'detected_key':     full['detected_key'],
            'key_confidence':   full['key_confidence'],
            'confidence':       0.65,
            'tracker':          'librosa_segmented',
        }

    def _analyze_segment(self, segment: np.ndarray) -> Tuple[float, float]:
        """Blocking helper — returns (tempo, swing_ratio) for one segment."""
        onset_low = librosa.onset.onset_strength(
            y=segment, sr=self.sr, aggregate=np.median, fmin=40, fmax=200)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_low, sr=self.sr, start_bpm=120.0)
        tempo      = float(np.atleast_1d(tempo)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        swing, _   = self._detect_swing_ratio(segment, beat_times, tempo)
        return tempo, swing

    # ── Swing detection ─────────────────────────────────────────────────────

    def _detect_swing_ratio(self, audio: np.ndarray,
                             beat_times: np.ndarray,
                             tempo: float) -> Tuple[float, float]:
        """
        Detect swing at the EIGHTH-NOTE subdivision level.

        The original code measured beat-to-beat intervals (quarter notes).
        That measures tempo stability, not swing.  Swing lives in how the
        two eighth notes within each beat are proportioned.

          Straight:     |---50%---|---50%---|  ratio = 1.0
          Medium swing: |---60%---|---40%---|  ratio = 1.5
          Hard swing:   |---67%---|---33%---|  ratio = 2.0  (triplet feel)

        Returns:
            (swing_ratio, confidence)
        """
        if len(beat_times) < 4:
            return 1.0, 0.2

        # Not enough beats for subdivision analysis — fall back to beat-level
        if len(beat_times) < 8:
            intervals = np.diff(beat_times)
            even_iv   = intervals[0::2]
            odd_iv    = intervals[1::2]
            if len(even_iv) == 0 or len(odd_iv) == 0:
                return 1.0, 0.2
            avg_e = np.mean(even_iv)
            avg_o = np.mean(odd_iv)
            if avg_e == 0 or avg_o == 0:
                return 1.0, 0.2
            ratio = float(min(2.0, max(0.5, max(avg_e, avg_o) / min(avg_e, avg_o))))
            return ratio, 0.3

        # Eighth-note subdivision via onset peaks
        beat_duration = 60.0 / max(tempo, 1.0)
        hop_length    = 512

        onset_env  = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=hop_length)
        frame_times = librosa.frames_to_time(
            np.arange(len(onset_env)), sr=self.sr, hop_length=hop_length)

        long_eighths:  list = []
        short_eighths: list = []

        for b in range(len(beat_times) - 1):
            b_start  = beat_times[b]
            b_end    = beat_times[b + 1]
            mid_pt   = b_start + beat_duration * 0.5

            mask1 = (frame_times >= b_start) & (frame_times < mid_pt)
            mask2 = (frame_times >= mid_pt)  & (frame_times < b_end)

            if mask1.any() and mask2.any():
                pk2_time = frame_times[mask2][np.argmax(onset_env[mask2])]
                long_dur  = pk2_time - b_start
                short_dur = b_end   - pk2_time
                if long_dur > 0 and short_dur > 0:
                    long_eighths.append(long_dur)
                    short_eighths.append(short_dur)

        if len(long_eighths) < 4:
            # Not enough subdivision data — beat-level fallback
            intervals = np.diff(beat_times)
            even_iv   = intervals[0::2]
            odd_iv    = intervals[1::2]
            if len(even_iv) and len(odd_iv):
                avg_e = np.mean(even_iv)
                avg_o = np.mean(odd_iv)
                if avg_e > 0 and avg_o > 0:
                    return float(min(2.0, max(0.5, max(avg_e, avg_o) / min(avg_e, avg_o)))), 0.3
            return 1.0, 0.2

        avg_long  = float(np.median(long_eighths))
        avg_short = float(np.median(short_eighths))
        if avg_short <= 0:
            return 1.0, 0.3

        ratio    = float(np.clip(avg_long / avg_short, 0.5, 2.5))
        var_comb = np.var(long_eighths) + np.var(short_eighths)
        consist  = 1.0 / (1.0 + var_comb * 20)
        conf     = float(np.clip(consist * (0.4 + abs(ratio - 1.0) * 2), 0.0, 0.95))

        return ratio, conf

    # ── Time signature ──────────────────────────────────────────────────────

    def _detect_time_signature(self, onset_env: np.ndarray,
                                beat_times: np.ndarray,
                                tempo: float) -> Tuple[str, list]:
        """
        Detect time signature via PLP.
        FIX: uses actual tempo instead of hardcoded 120 BPM.
        """
        if len(beat_times) < 8:
            fallback = beat_times[::4].tolist() if len(beat_times) >= 4 else beat_times.tolist()
            return "4/4", fallback

        hop_length = 512
        try:
            pulse  = librosa.beat.plp(onset_envelope=onset_env, sr=self.sr,
                                       hop_length=hop_length)
            peaks, _ = find_peaks(pulse, height=np.mean(pulse) + np.std(pulse))

            if len(peaks) > 3:
                frame_dur_s   = hop_length / self.sr
                beat_dur_s    = 60.0 / max(tempo, 1.0)   # FIX: actual tempo
                frames_per_beat = beat_dur_s / frame_dur_s

                periods = np.diff(peaks)
                if len(periods) > 0:
                    median_period = float(np.median(periods))
                    bpb = int(np.clip(round(median_period / frames_per_beat), 2, 7))
                    ts_map    = {2: "2/4", 3: "3/4", 4: "4/4",
                                 5: "5/4", 6: "6/8", 7: "7/8"}
                    time_sig  = ts_map.get(bpb, "4/4")
                    downbeats = librosa.frames_to_time(
                        peaks, sr=self.sr, hop_length=hop_length).tolist()
                    return time_sig, downbeats

        except Exception as e:
            print(f"   ⚠️ PLP time-signature detection failed: {e}")

        fallback = beat_times[::4].tolist() if len(beat_times) >= 4 else beat_times.tolist()
        return "4/4", fallback

    # ── Key detection ───────────────────────────────────────────────────────

    def _detect_key(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Detect musical key via CQT chroma + Krumhansl-Kessler profiles.
        FIX: now checks all 24 keys (12 major + 12 minor).
        The original only checked major — missing most jazz tonalities.
        """
        try:
            chroma   = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
            chroma_m = np.mean(chroma, axis=1)
            chroma_m = chroma_m / (chroma_m.sum() + 1e-8)

            best_key   = 'C Major'
            best_score = -np.inf

            for key_name, template in _KEY_TEMPLATES.items():
                score = float(np.dot(chroma_m, template))
                if score > best_score:
                    best_score = score
                    best_key   = key_name

            # Normalise: dot product of normalised vectors ≈ 0.06–0.20
            confidence = float(np.clip((best_score - 0.06) / (0.20 - 0.06), 0.3, 0.95))
            return best_key, confidence

        except Exception as e:
            print(f"   ⚠️ Key detection failed: {e}")
            return "C Major", 0.3

    # ── Confidence ──────────────────────────────────────────────────────────

    def _calculate_confidence(self, tempo: float, beat_times: np.ndarray,
                               swing_ratio: float,
                               key_confidence: float) -> float:
        """Overall confidence (0–1) for ConfidenceRouter."""
        conf = 0.45
        if 50 < tempo < 260:
            conf += 0.10
        n = len(beat_times)
        if n > 32:
            conf += 0.18
        elif n > 16:
            conf += 0.12
        elif n > 8:
            conf += 0.06
        else:
            conf -= 0.10   # almost no beats — penalise hard
        if 0.75 < swing_ratio < 2.2:
            conf += 0.07
        conf += key_confidence * 0.10
        return float(np.clip(conf, 0.0, 0.92))


# ── Singleton ────────────────────────────────────────────────────────────────

_librosa_tracker: Optional[LibrosaBeatIntelligence] = None


def get_librosa_tracker() -> LibrosaBeatIntelligence:
    global _librosa_tracker
    if _librosa_tracker is None:
        _librosa_tracker = LibrosaBeatIntelligence()
    return _librosa_tracker


# ============================================================================
# PIPELINE INTEGRATION — called from madmom.py when Madmom fails
# ============================================================================

async def enrich_rhythm_info_librosa(rhythm_info,
                                      audio: np.ndarray,
                                      sr: int,
                                      user_tempo: float = None) -> None:
    """
    Drop-in replacement for madmom.enrich_rhythm_info() when Madmom
    times out.

    HOW TO USE IN madmom.py:

        async def enrich_rhythm_info(rhythm_info, audio, sr, user_tempo=None):
            tracker = get_madmom_tracker()
            result  = await tracker.track(audio, sr)

            if result is None:
                # Madmom failed — use high-quality librosa fallback
                from separation.librosa_tracker import enrich_rhythm_info_librosa
                await enrich_rhythm_info_librosa(rhythm_info, audio, sr, user_tempo)
                return

            # ... rest of madmom code unchanged ...

    Writes in-place onto rhythm_info so nothing else in the pipeline changes.
    """
    tracker = get_librosa_tracker()
    result  = await tracker.track(audio, sr)

    if result is None:
        rhythm_info.tracker_source = 'librosa_fallback'
        return

    # Core beat fields
    if result.get('beats'):
        rhythm_info.beat_times = result['beats']
    if result.get('downbeats'):
        rhythm_info.downbeats = result['downbeats']
    if result.get('tempo', 0) > 20:
        rhythm_info.tempo = result['tempo']

    rhythm_info.confidence     = result.get('confidence', 0.5)
    rhythm_info.tracker_source = result.get('tracker', 'librosa_fallback')

    ts_str = result.get('time_signature', '4/4')
    rhythm_info.time_signature = ts_str
    try:
        rhythm_info.beats_per_bar = int(ts_str.split('/')[0])
    except Exception:
        rhythm_info.beats_per_bar = 4

    # Swing fields (written if RhythmInfo has them — add to order_types.py)
    if hasattr(rhythm_info, 'swing_ratio'):
        rhythm_info.swing_ratio      = result.get('swing_ratio', 1.0)
        rhythm_info.swing_confidence = result.get('swing_confidence', 0.5)

    # Key fields (written if RhythmInfo has them — add to order_types.py)
    if hasattr(rhythm_info, 'detected_key'):
        rhythm_info.detected_key   = result.get('detected_key', 'C Major')
        rhythm_info.key_confidence = result.get('key_confidence', 0.3)

    # Rebuild eighth-note grid with swing-aware spacing
    if rhythm_info.beat_times:
        beat_dur = 60.0 / max(rhythm_info.tempo, 1.0)
        swing    = getattr(rhythm_info, 'swing_ratio', 1.0)
        total    = swing + 1.0
        grid     = []
        for bt in rhythm_info.beat_times:
            grid.append(round(bt, 4))
            # Long eighth position differs when swung
            if swing > 1.05:
                grid.append(round(bt + beat_dur * swing / total, 4))
            else:
                grid.append(round(bt + beat_dur * 0.5, 4))
        rhythm_info.grid = sorted(grid)

    # user_tempo deviation warning (mirrors madmom.py behaviour)
    if user_tempo and user_tempo > 0:
        dev = abs(rhythm_info.tempo - user_tempo)
        if dev > 5.0:
            print(f"   ⚠️ LibrosaTracker: detected {rhythm_info.tempo:.1f} BPM "
                  f"deviates {dev:.1f} BPM from guided tempo {user_tempo:.1f} BPM")

    swing_label = "swung 🎷" if getattr(rhythm_info, 'swing_ratio', 1.0) > 1.15 else "straight"
    print(f"   🎵 LibrosaTracker: {rhythm_info.tempo:.1f} BPM | "
          f"{rhythm_info.time_signature} | {swing_label} | "
          f"key={getattr(rhythm_info, 'detected_key', '?')} | "
          f"confidence={rhythm_info.confidence:.2f}")


async def get_swing_ratio(audio: np.ndarray, sr: int) -> float:
    """Quick helper — returns just the swing ratio."""
    return (await get_librosa_tracker().track(audio, sr)).get('swing_ratio', 1.0)


async def get_rhythm_profile(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Full rhythm profile dict for external callers."""
    return await get_librosa_tracker().track(audio, sr)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Librosa Beat Intelligence")
    parser.add_argument("audio_file", help="Path to audio file")
    args = parser.parse_args()

    async def _test():
        audio, sr = librosa.load(args.audio_file, sr=22050)
        result = await get_librosa_tracker().track(audio, sr)

        print("\n" + "=" * 60)
        print("LIBROSA BEAT INTELLIGENCE OUTPUT")
        print("=" * 60)
        print(f"Tempo:            {result['tempo']:.1f} BPM")
        print(f"Time Signature:   {result['time_signature']}")
        print(f"Beats detected:   {len(result['beats'])}")
        print(f"Downbeats:        {len(result['downbeats'])}")
        print(f"Key:              {result['detected_key']}  "
              f"(conf {result.get('key_confidence', 0):.2f})")
        print(f"Swing Ratio:      {result['swing_ratio']:.3f}  "
              f"(1.0=straight, ~1.5=jazz, ~2.0=hard swing)")
        print(f"Swing Confidence: {result.get('swing_confidence', 0):.2f}")
        print(f"Overall Conf:     {result['confidence']:.2f}")
        print(f"Tracker:          {result['tracker']}")

        sr_val = result['swing_ratio']
        if   sr_val > 1.5:  print("\n🎷 HARD SWING  — Classic bebop/jazz feel")
        elif sr_val > 1.2:  print("\n🎵 MEDIUM SWING — Relaxed jazz feel")
        elif sr_val > 1.08: print("\n🎶 LIGHT SWING  — Slight shuffle")
        else:               print("\n📏 STRAIGHT     — Even eighth notes")

    asyncio.run(_test())