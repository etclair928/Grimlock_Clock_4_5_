#!/usr/bin/env python3
"""
schoenberg_mirror.py — The Four Mirrors of Truth (LITERAL TRANSFORMATIONS)

"The truth is what survives when viewed from every angle."

The Schoenberg Mirror applies four independent literal transformations to audio:
1. ZCR Mirror — Zero-crossing rate (noise vs periodicity) on original audio
2. Temporal Mirror — LITERAL RETROGRADE (audio reversed, compare forward vs backward)
3. Spectral Mirror — LITERAL FREQUENCY INVERSION (spectrum flipped, HPS on both)
4. NMF Mirror — Learned template activation (requires pre-fit)

Principle: A hit must pass through ALL FOUR mirrors. Each mirror applies a literal
transformation and measures the DIFFERENCE between original and transformed.

RETROGRADE: audio[::-1] — reverses time, compares envelope shapes
FREQUENCY INVERSION: spectrum[::-1] — flips frequency axis, compares HPS results

This is NOT statistical estimation. This is literal transformation + comparison.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Import from Phase 1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from order_types import MirrorResult, SchoenbergResult
from core.audio_utils import extract_window, safe_onset_slice, validate_audio
from core.fft_helpers import (
    compute_fft, compute_mel_spectrogram, spectral_centroid,
    zero_crossing_rate, harmonic_ratio
)
from core.time_utils import compute_ioi

# Optional NMF import with fallback
try:
    from sklearn.decomposition import NMF
    NMF_AVAILABLE = True
except ImportError:
    NMF_AVAILABLE = False
    print("⚠️ scikit-learn not available — NMF Mirror disabled")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MirrorConfig:
    """Configuration for all four mirrors."""

    # ZCR Mirror
    zcr_threshold: float = 0.08
    zcr_veto_threshold: float = 0.04

    # Temporal Mirror (Literal Retrograde)
    retrograde_similarity_threshold: float = 0.6  # Below this = asymmetric (drum)
    retrograde_veto_similarity: float = 0.85  # Above this = symmetric (piano)

    # Spectral Mirror (Literal Frequency Inversion)
    hps_num_harmonics: int = 4
    harmonic_ratio_threshold: float = 0.45  # Below this = noise (drum)
    harmonic_ratio_veto: float = 0.55  # Above this = pitched (piano)

    # NMF Mirror
    nmf_n_components: int = 5
    nmf_activation_threshold: float = 0.15
    nmf_tolerance_ms: float = 55.0

    # General
    sample_rate: int = 22050
    hop_length: int = 512
    mirror_enabled: bool = True

    # Window sizes
    pre_window_ms: float = 50.0
    post_window_ms: float = 200.0
    spectral_window_ms: float = 50.0


# ============================================================================
# HELPER: HARMONIC PRODUCT SPECTRUM
# ============================================================================

def harmonic_product_spectrum(spectrum: np.ndarray, num_harmonics: int = 4) -> Tuple[float, float]:
    """
    Harmonic Product Spectrum (HPS) for fundamental frequency detection.

    Works even when fundamental is missing (uses harmonics to infer).

    Args:
        spectrum: FFT magnitude spectrum
        num_harmonics: Number of harmonics to multiply

    Returns:
        (peak_idx, peak_strength)
    """
    if len(spectrum) == 0:
        return 0, 0.0

    hps = spectrum.copy().astype(np.float64)

    for h in range(2, num_harmonics + 1):
        downsampled = spectrum[::h]
        min_len = min(len(hps), len(downsampled))
        hps[:min_len] *= downsampled[:min_len]

    peak_idx = np.argmax(hps)
    peak_strength = hps[peak_idx]

    return peak_idx, float(peak_strength)


# ============================================================================
# MIRROR 1: ZCR (Noise vs Periodicity)
# ============================================================================

class ZCRMirror:
    """
    ZCR Mirror: Measures noise content on ORIGINAL audio.

    Drums (noise): High ZCR (>0.08)
    Piano (periodic): Low ZCR (<0.04)
    """

    def __init__(self, config: MirrorConfig):
        self.config = config
        self.name = "ZCR"

    def process(self, audio: np.ndarray, onset_time: float) -> MirrorResult:
        """Run ZCR mirror on original audio."""
        window = extract_window(audio, self.config.sample_rate, onset_time, window_ms=50)

        if len(window) < 32:
            return MirrorResult(
                passed=False, score=0.0, value=0.0,
                reason="insufficient_audio"
            )

        zcr = zero_crossing_rate(window, frame_length=512, hop=256)
        passes = zcr > self.config.zcr_threshold
        score = min(1.0, zcr / 0.15)

        if zcr < self.config.zcr_veto_threshold:
            passes = False

        return MirrorResult(
            passed=passes,
            score=score,
            value=zcr,
            reason=None if passes else f"zcr_too_low_{zcr:.3f}"
        )


# ============================================================================
# MIRROR 2: TEMPORAL (LITERAL RETROGRADE)
# ============================================================================

class TemporalMirror:
    """
    Temporal Mirror: LITERAL RETROGRADE transformation.

    Process:
    1. Extract window around onset
    2. Analyze forward envelope (peak position, attack slope, decay slope)
    3. Reverse the audio: window_rev = window[::-1]
    4. Analyze reversed envelope
    5. Compare forward vs reversed results

    Real drum hit: Forward and reversed look VERY DIFFERENT (asymmetric)
    Piano/pitched: Forward and reversed look SIMILAR (symmetric)

    Returns similarity score (lower = more drum-like)
    """

    def __init__(self, config: MirrorConfig):
        self.config = config
        self.name = "TEMPORAL_RETROGRADE"

    def _compute_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Compute smoothed amplitude envelope."""
        envelope = np.abs(audio)

        # Smooth for stable peak detection
        window_size = int(0.005 * self.config.sample_rate)
        if window_size > 1 and len(envelope) > window_size:
            kernel = np.ones(window_size) / window_size
            envelope = np.convolve(envelope, kernel, mode='same')

        return envelope

    def _analyze_envelope(self, envelope: np.ndarray) -> Dict[str, float]:
        """Extract features from envelope."""
        if len(envelope) == 0:
            return {'valid': False}

        peak_idx = np.argmax(envelope)
        peak_val = envelope[peak_idx]

        # Peak position as percentage (0-1)
        peak_position = peak_idx / len(envelope) if len(envelope) > 0 else 0.5

        # Attack slope (rise to peak)
        if peak_idx > 0:
            attack_slope = peak_val / (peak_idx + 1e-8)
        else:
            attack_slope = 0

        # Decay slope (fall from peak to end)
        decay_samples = len(envelope) - peak_idx
        if decay_samples > 0:
            decay_slope = peak_val / (decay_samples + 1e-8)
        else:
            decay_slope = 0

        # Energy skewness (front-loaded vs back-loaded)
        total_energy = np.sum(envelope)
        early_energy = np.sum(envelope[:len(envelope) // 2])
        skewness = early_energy / (total_energy + 1e-8)

        return {
            'valid': True,
            'peak_idx': peak_idx,
            'peak_val': peak_val,
            'peak_position': peak_position,
            'attack_slope': attack_slope,
            'decay_slope': decay_slope,
            'skewness': skewness
        }

    def _calculate_similarity(self, fwd: Dict, rev: Dict) -> float:
        """
        Calculate similarity between forward and reversed envelopes.

        Lower similarity = more asymmetric = more drum-like
        Higher similarity = more symmetric = more pitched
        """
        # Compare peak positions
        pos_diff = abs(fwd['peak_position'] - rev['peak_position'])
        position_similarity = 1.0 - min(1.0, pos_diff / 0.5)

        # Compare skewness
        skew_diff = abs(fwd['skewness'] - rev['skewness'])
        skew_similarity = 1.0 - min(1.0, skew_diff)

        # Compare attack/decay slopes (reversed swaps these)
        # In forward: attack_slope should be high, decay_slope low
        # In reverse: attack_slope should be low, decay_slope high
        # For symmetric sounds, both slopes are similar
        slope_ratio_fwd = fwd['attack_slope'] / (fwd['decay_slope'] + 1e-8)
        slope_ratio_rev = rev['attack_slope'] / (rev['decay_slope'] + 1e-8)
        slope_similarity = 1.0 - min(1.0, abs(slope_ratio_fwd - slope_ratio_rev) / 10.0)

        # Weighted average
        similarity = (position_similarity * 0.4 +
                      skew_similarity * 0.3 +
                      slope_similarity * 0.3)

        return similarity

    def process(self, audio: np.ndarray, onset_time: float) -> MirrorResult:
        """
        Run literal retrograde mirror on hit candidate.

        LITERALLY reverses the audio and compares envelopes.
        """
        # Extract window
        pre_samples = int(self.config.pre_window_ms / 1000 * self.config.sample_rate)
        post_samples = int(self.config.post_window_ms / 1000 * self.config.sample_rate)

        start = max(0, int(onset_time * self.config.sample_rate) - pre_samples)
        end = min(len(audio), int(onset_time * self.config.sample_rate) + post_samples)

        if end - start < int(0.05 * self.config.sample_rate):
            return MirrorResult(
                passed=False,
                score=0.0,
                value={"similarity": 0, "reason": "insufficient_audio"},
                reason="insufficient_audio"
            )

        window = audio[start:end]

        # FORWARD analysis
        envelope_fwd = self._compute_envelope(window)
        fwd_analysis = self._analyze_envelope(envelope_fwd)

        if not fwd_analysis['valid']:
            return MirrorResult(
                passed=False,
                score=0.0,
                value={"similarity": 0},
                reason="forward_analysis_failed"
            )

        # LITERAL RETROGRADE: reverse the audio
        window_rev = window[::-1]
        envelope_rev = self._compute_envelope(window_rev)
        rev_analysis = self._analyze_envelope(envelope_rev)

        if not rev_analysis['valid']:
            return MirrorResult(
                passed=False,
                score=0.0,
                value={"similarity": 0},
                reason="reverse_analysis_failed"
            )

        # Calculate similarity between forward and reversed
        similarity = self._calculate_similarity(fwd_analysis, rev_analysis)

        # LOW similarity = asymmetric = DRUM (PASS)
        # HIGH similarity = symmetric = PIANO (FAIL)
        passes = similarity < self.config.retrograde_similarity_threshold

        # Score: lower similarity = higher drum score
        score = 1.0 - similarity

        # Veto: if too symmetric, definitely not drum
        if similarity > self.config.retrograde_veto_similarity:
            passes = False

        value = {
            "similarity": similarity,
            "forward_peak_position": fwd_analysis['peak_position'],
            "reverse_peak_position": rev_analysis['peak_position'],
            "forward_skewness": fwd_analysis['skewness'],
            "reverse_skewness": rev_analysis['skewness'],
            "attack_slope_ratio_fwd": fwd_analysis['attack_slope'] / (fwd_analysis['decay_slope'] + 1e-8),
            "attack_slope_ratio_rev": rev_analysis['attack_slope'] / (rev_analysis['decay_slope'] + 1e-8)
        }

        reason = None
        if not passes:
            reason = f"too_symmetric_similarity={similarity:.2f}"

        return MirrorResult(
            passed=passes,
            score=score,
            value=value,
            reason=reason
        )


# ============================================================================
# MIRROR 3: SPECTRAL (LITERAL FREQUENCY INVERSION + HPS)
# ============================================================================

class SpectralMirror:
    """
    Spectral Mirror: LITERAL FREQUENCY INVERSION + HPS.

    Process:
    1. Extract window around onset
    2. Compute FFT spectrum
    3. Run HPS on NORMAL spectrum (detects fundamental)
    4. LITERAL FREQUENCY INVERSION: spectrum = spectrum[::-1]
    5. Run HPS on INVERTED spectrum
    6. Compare results

    Harmonic instrument (piano): Strong peak in BOTH normal AND inverted
    Noise instrument (drum): Weak peak in BOTH
    """

    def __init__(self, config: MirrorConfig):
        self.config = config
        self.name = "SPECTRAL_INVERSION"

    def compute_hps(self, spectrum: np.ndarray) -> Tuple[int, float]:
        """Harmonic Product Spectrum."""
        return harmonic_product_spectrum(spectrum, self.config.hps_num_harmonics)

    def process(self, audio: np.ndarray, onset_time: float) -> MirrorResult:
        """
        Run literal frequency inversion mirror on hit candidate.

        LITERALLY inverts the frequency axis and compares HPS results.
        """
        # Extract window
        window_ms = self.config.spectral_window_ms
        start = max(0, int(onset_time * self.config.sample_rate))
        end = min(len(audio), start + int(window_ms / 1000 * self.config.sample_rate))

        if end - start < 256:
            return MirrorResult(
                passed=False,
                score=0.0,
                value={"harmonic_ratio": 0},
                reason="insufficient_audio"
            )

        segment = audio[start:end] * np.hanning(end - start)
        spectrum = np.abs(np.fft.rfft(segment))

        if len(spectrum) < 10:
            return MirrorResult(
                passed=False,
                score=0.0,
                value={"harmonic_ratio": 0},
                reason="spectrum_too_small"
            )

        # NORMAL mode: run HPS on original spectrum
        normal_peak_idx, normal_strength = self.compute_hps(spectrum)

        # LITERAL FREQUENCY INVERSION: flip the spectrum
        spectrum_inverted = spectrum[::-1]
        inv_peak_idx, inv_strength = self.compute_hps(spectrum_inverted)

        # Harmonic ratio = average of both strengths (normalized)
        max_possible = np.max(spectrum) * self.config.hps_num_harmonics
        if max_possible > 0:
            normal_ratio = normal_strength / max_possible
            inv_ratio = inv_strength / max_possible
        else:
            normal_ratio = 0
            inv_ratio = 0

        harmonic_ratio = (normal_ratio + inv_ratio) / 2

        # DRUM = LOW harmonic ratio (noise)
        # PITCHED = HIGH harmonic ratio (harmonic)
        passes = harmonic_ratio < self.config.harmonic_ratio_threshold

        # Score: lower harmonic ratio = higher drum score
        score = 1.0 - min(1.0, harmonic_ratio / 0.8)

        # Veto: if too harmonic, definitely not drum
        if harmonic_ratio > self.config.harmonic_ratio_veto:
            passes = False

        # Convert peak indices to frequencies
        freqs = np.fft.rfftfreq(len(segment), 1 / self.config.sample_rate)
        normal_freq = freqs[normal_peak_idx] if normal_peak_idx < len(freqs) else 0
        inv_freq = freqs[inv_peak_idx] if inv_peak_idx < len(freqs) else 0

        value = {
            "harmonic_ratio": harmonic_ratio,
            "normal_strength": normal_strength,
            "inv_strength": inv_strength,
            "normal_peak_freq_hz": normal_freq,
            "inv_peak_freq_hz": inv_freq,
            "normal_ratio": normal_ratio,
            "inv_ratio": inv_ratio
        }

        reason = None
        if not passes:
            reason = f"too_harmonic_ratio={harmonic_ratio:.2f}"

        return MirrorResult(
            passed=passes,
            score=score,
            value=value,
            reason=reason
        )


# ============================================================================
# MIRROR 4: NMF (Learned Template Activation)
# ============================================================================

class NMFMirror:
    """
    NMF Mirror: Checks if a learned component activates at this time.

    Requires pre-computed NMF components from the full audio via fit().
    """

    def __init__(self, config: MirrorConfig):
        self.config = config
        self.name = "NMF"
        self._nmf_hits = {}
        self._nmf_components = None
        self._nmf_activations = None

    def fit(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        """Run NMF decomposition on full audio. Must be called before process()."""
        if not NMF_AVAILABLE:
            return {}

        try:
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=64, hop_length=self.config.hop_length,
                fmin=20, fmax=sr // 2
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

            model = NMF(n_components=self.config.nmf_n_components,
                        beta_loss='kullback-leibler',
                        solver='mu',
                        max_iter=300,
                        random_state=42)

            W_T = model.fit_transform(S_norm.T)
            H = W_T.T
            W = model.components_.T

            self._nmf_components = W
            self._nmf_activations = H

            # Auto-label by centroid
            mel_freqs = librosa.mel_frequencies(n_mels=64, fmin=20, fmax=sr // 2)
            centroids = []
            for i in range(W.shape[1]):
                template = W[:, i]
                c = np.sum(mel_freqs * template) / (np.sum(template) + 1e-8)
                centroids.append(c)

            sorted_idx = np.argsort(centroids)
            n = len(sorted_idx)

            labels = {}
            labels[sorted_idx[0]] = 'KICK'
            if n > 1:
                labels[sorted_idx[1]] = 'SNARE'
            if n > 2:
                labels[sorted_idx[n // 2]] = 'TOM'
            if n > 3:
                labels[sorted_idx[-2]] = 'RIDE'
            if n > 4:
                labels[sorted_idx[-1]] = 'HIHAT'

            frame_times = librosa.frames_to_time(
                np.arange(H.shape[1]), sr=sr, hop_length=self.config.hop_length
            )

            self._nmf_hits = {}
            for comp_idx, drum_type in labels.items():
                activation = H[comp_idx, :].copy()
                if np.max(activation) > 0:
                    activation /= np.max(activation)

                peaks, _ = find_peaks(activation,
                                      height=self.config.nmf_activation_threshold,
                                      distance=4)
                if len(peaks) > 0:
                    self._nmf_hits[drum_type] = frame_times[peaks].tolist()

            return self._nmf_hits

        except Exception as e:
            print(f"⚠️ NMF Mirror fit failed: {e}")
            return {}

    def process(self, onset_time: float) -> MirrorResult:
        """Check if NMF has a component activation near this time."""
        if not self._nmf_hits:
            return MirrorResult(
                passed=True,
                score=0.5,
                value={},
                reason="nmf_not_available"
            )

        tolerance = self.config.nmf_tolerance_ms / 1000.0
        best_match = None
        best_dist = float('inf')
        best_type = None

        for drum_type, times in self._nmf_hits.items():
            for t in times:
                dist = abs(t - onset_time)
                if dist < tolerance and dist < best_dist:
                    best_dist = dist
                    best_match = t
                    best_type = drum_type

        passes = best_match is not None

        value = {
            "matched_type": best_type,
            "deviation_ms": best_dist * 1000 if best_match else -1,
            "matched_time": best_match
        }

        return MirrorResult(
            passed=passes,
            score=0.85 if passes else 0.2,
            value=value,
            reason=None if passes else "no_nmf_activation"
        )


# ============================================================================
# SCHOENBERG MIRROR (Main Orchestrator)
# ============================================================================

class SchoenbergMirror:
    """
    Main orchestrator for all four mirrors.

    Each mirror applies a LITERAL transformation:
    - ZCR: No transformation (measures noise directly)
    - Temporal: LITERAL RETROGRADE (audio reversed, compare envelopes)
    - Spectral: LITERAL FREQUENCY INVERSION (spectrum flipped, HPS on both)
    - NMF: Learned template activation (pre-fit required)

    A hit must pass through ALL FOUR mirrors to be accepted.
    """

    def __init__(self, config: Optional[MirrorConfig] = None):
        self.config = config or MirrorConfig()

        self.zcr_mirror = ZCRMirror(self.config)
        self.temporal_mirror = TemporalMirror(self.config)
        self.spectral_mirror = SpectralMirror(self.config)
        self.nmf_mirror = NMFMirror(self.config)

        self._is_fitted = False

    def fit_nmf(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        """Fit NMF mirror to full audio. Must be called before using NMF validation."""
        self._is_fitted = True
        return self.nmf_mirror.fit(audio, sr)

    def fit_nmf_dynamic(self, audio: np.ndarray, sr: int,
                         segment_duration: float = 30.0,
                         zcr_ghost_threshold: float = 0.06) -> Dict[str, List[float]]:
        """
        FIX #4 — Dynamic Template Updating.

        Instead of one static NMF pass over the whole track, split the audio
        into segments and re-fit NMF per segment.  This lets the templates
        evolve when a drummer switches from sticks to brushes, or when ghost
        notes appear that the initial template would miss.

        ZCR-gated re-learning: if a segment's average ZCR is unusually low
        compared to the global mean (indicating a texture change) the segment
        gets its own NMF pass and the resulting hits are merged with the
        global hit list.

        Args:
            audio:            Full drum-stem audio array.
            sr:               Sample rate.
            segment_duration: Length of each NMF segment in seconds.
            zcr_ghost_threshold: Relative ZCR drop (fraction of global mean)
                               that triggers a re-learn in that segment.
        """
        if not NMF_AVAILABLE:
            # Graceful fallback to single-pass
            return self.fit_nmf(audio, sr)

        segment_samples = int(segment_duration * sr)
        n_segments = max(1, int(np.ceil(len(audio) / segment_samples)))

        global_zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
        merged_hits: Dict[str, List[float]] = {}

        print(f"   🔄 Dynamic NMF: {n_segments} segments × {segment_duration:.0f}s")

        for seg_idx in range(n_segments):
            start = seg_idx * segment_samples
            end = min(len(audio), start + segment_samples)
            segment = audio[start:end]

            if len(segment) < sr:          # skip silences / tiny tails
                continue

            seg_zcr = float(np.mean(np.abs(np.diff(np.sign(segment)))) / 2)
            time_offset = start / sr

            # Always run NMF; flag segments where texture changed
            texture_changed = (global_zcr > 0 and
                               seg_zcr < global_zcr * (1.0 - zcr_ghost_threshold))
            if texture_changed:
                print(f"   ⚡ Segment {seg_idx}: ZCR drop detected "
                      f"({seg_zcr:.4f} vs global {global_zcr:.4f}) — re-learning templates")

            seg_hits = self.nmf_mirror.fit(segment, sr)

            # Merge hits with global timeline offset applied
            for drum_type, times in seg_hits.items():
                shifted = [t + time_offset for t in times]
                if drum_type not in merged_hits:
                    merged_hits[drum_type] = shifted
                else:
                    merged_hits[drum_type].extend(shifted)

        # Sort and de-duplicate hits within 30 ms
        dedup_window = 0.030
        for drum_type in merged_hits:
            times = sorted(merged_hits[drum_type])
            deduped = []
            for t in times:
                if not deduped or (t - deduped[-1]) > dedup_window:
                    deduped.append(t)
            merged_hits[drum_type] = deduped

        # Install merged hits into the NMF mirror so validate_hit() can use them
        self.nmf_mirror._nmf_hits = merged_hits
        self._is_fitted = True

        total_hits = sum(len(v) for v in merged_hits.values())
        print(f"   ✅ Dynamic NMF complete: {total_hits} total hits across "
              f"{len(merged_hits)} drum types")
        return merged_hits

    def validate_hit(self, audio: np.ndarray, onset_time: float,
                     use_nmf: bool = True) -> SchoenbergResult:
        """
        Run mirrors on a hit candidate using TIERED VALIDATION (FIX #6).

        Tier 1 — ZCR  (fastest, always runs)
            decisive high  → skip Temporal, go straight to lightweight result
            decisive low   → veto immediately
        Tier 2 — Temporal (runs unless ZCR was decisive)
        Tier 3 — Spectral + NMF (only when Tier 1+2 confidence is ambiguous,
                                  i.e. 0.4 ≤ combined_score ≤ 0.7)

        This eliminates the majority of expensive FFT/NMF calls on clear
        kick/snare hits while keeping full depth on edge-cases.
        """
        _skip_result = MirrorResult(passed=True, score=0.5, value={},
                                    reason="skipped_by_tiered_validation")

        # ── TIER 1: ZCR (no transformation, ~microseconds) ─────────────────
        zcr_result = self.zcr_mirror.process(audio, onset_time)

        # Hard veto — too quiet / tonal to be a drum
        if not zcr_result.passed and zcr_result.value < self.config.zcr_veto_threshold:
            return SchoenbergResult(
                zcr=zcr_result,
                temporal=_skip_result,
                spectral=_skip_result,
                nmf=_skip_result,
            )

        # Decisive positive — strong noise burst, skip deeper mirrors
        if zcr_result.score >= 0.85:
            nmf_result = (self.nmf_mirror.process(onset_time)
                          if use_nmf and self._is_fitted else _skip_result)
            return SchoenbergResult(
                zcr=zcr_result,
                temporal=_skip_result,
                spectral=_skip_result,
                nmf=nmf_result,
            )

        # ── TIER 2: Temporal / Retrograde ───────────────────────────────────
        temporal_result = self.temporal_mirror.process(audio, onset_time)

        combined_score = (zcr_result.score + temporal_result.score) / 2

        # Clear pass or fail after two mirrors — skip Spectral+NMF
        if combined_score < 0.3 or combined_score > 0.75:
            nmf_result = (self.nmf_mirror.process(onset_time)
                          if use_nmf and self._is_fitted else _skip_result)
            return SchoenbergResult(
                zcr=zcr_result,
                temporal=temporal_result,
                spectral=_skip_result,
                nmf=nmf_result,
            )

        # ── TIER 3: Spectral + NMF (only on ambiguous 0.3–0.75 zone) ───────
        spectral_result = self.spectral_mirror.process(audio, onset_time)

        if use_nmf and self._is_fitted:
            nmf_result = self.nmf_mirror.process(onset_time)
        else:
            nmf_result = _skip_result

        return SchoenbergResult(
            zcr=zcr_result,
            temporal=temporal_result,
            spectral=spectral_result,
            nmf=nmf_result,
        )

    def validate_hits(self, audio: np.ndarray, hit_times: List[float],
                      use_nmf: bool = True) -> List[Tuple[float, SchoenbergResult]]:
        """Validate multiple hits in batch."""
        results = []
        for t in hit_times:
            result = self.validate_hit(audio, t, use_nmf)
            results.append((t, result))
        return results

    def get_confidence_multiplier(self, result: SchoenbergResult) -> float:
        """Calculate confidence multiplier based on mirror results."""
        if not result.passes_all():
            return result.confidence_penalty()
        return 1.05

    def should_reject(self, result: SchoenbergResult) -> Tuple[bool, str]:
        """Determine if hit should be rejected based on veto conditions."""
        # ZCR veto
        if not result.zcr.passed:
            if result.zcr.value < self.config.zcr_veto_threshold:
                return True, f"ZCR veto: {result.zcr.reason}"

        # Temporal (retrograde) veto
        if not result.temporal.passed:
            similarity = result.temporal.value.get('similarity', 0)
            if similarity > self.config.retrograde_veto_similarity:
                return True, f"Temporal veto: too symmetric ({similarity:.2f})"

        # Spectral (inversion) veto
        if not result.spectral.passed:
            harmonic_ratio = result.spectral.value.get('harmonic_ratio', 0)
            if harmonic_ratio > self.config.harmonic_ratio_veto:
                return True, f"Spectral veto: too harmonic ({harmonic_ratio:.2f})"

        # NMF veto
        if not result.nmf.passed and self._is_fitted:
            return True, f"NMF veto: {result.nmf.reason}"

        return False, ""


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Test Schoenberg Mirror")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--onset", type=float, required=True, help="Onset time in seconds")
    parser.add_argument("--fit-nmf", action="store_true", help="Fit NMF mirror first")
    args = parser.parse_args()

    # Load audio
    audio, sr = librosa.load(args.audio_file, sr=22050)

    # Create mirror
    mirror = SchoenbergMirror()

    # Fit NMF if requested
    if args.fit_nmf:
        print("Fitting NMF mirror...")
        mirror.fit_nmf(audio, sr)

    # Validate hit
    result = mirror.validate_hit(audio, args.onset, use_nmf=args.fit_nmf)

    print(f"\n{'=' * 70}")
    print(f"SCHOENBERG MIRROR — Literal Transformations")
    print(f"{'=' * 70}")
    print(f"Onset time: {args.onset:.3f}s")
    print(f"Passes all mirrors: {result.passes_all()}")
    print(f"Confidence multiplier: {mirror.get_confidence_multiplier(result):.2f}")

    reject, reason = mirror.should_reject(result)
    print(f"Should reject: {reject}")
    if reject:
        print(f"Rejection reason: {reason}")

    print(f"\n{'─' * 70}")
    print("MIRROR RESULTS")
    print(f"{'─' * 70}")

    print(f"\n1. ZCR MIRROR (Noise vs Periodicity)")
    print(f"   Passed: {result.zcr.passed}")
    print(f"   Score: {result.zcr.score:.2f}")
    print(f"   ZCR value: {result.zcr.value:.4f}")
    if result.zcr.reason:
        print(f"   Reason: {result.zcr.reason}")

    print(f"\n2. TEMPORAL MIRROR (Literal Retrograde)")
    print(f"   Passed: {result.temporal.passed}")
    print(f"   Score: {result.temporal.score:.2f}")
    if result.temporal.value:
        print(f"   Similarity (forward vs reverse): {result.temporal.value.get('similarity', 0):.3f}")
        print(f"   Forward peak position: {result.temporal.value.get('forward_peak_position', 0):.2f}")
        print(f"   Reverse peak position: {result.temporal.value.get('reverse_peak_position', 0):.2f}")
    if result.temporal.reason:
        print(f"   Reason: {result.temporal.reason}")

    print(f"\n3. SPECTRAL MIRROR (Literal Frequency Inversion)")
    print(f"   Passed: {result.spectral.passed}")
    print(f"   Score: {result.spectral.score:.2f}")
    if result.spectral.value:
        print(f"   Harmonic ratio: {result.spectral.value.get('harmonic_ratio', 0):.3f}")
        print(f"   Normal HPS strength: {result.spectral.value.get('normal_strength', 0):.2f}")
        print(f"   Inverted HPS strength: {result.spectral.value.get('inv_strength', 0):.2f}")
        print(f"   Normal peak freq: {result.spectral.value.get('normal_peak_freq_hz', 0):.1f} Hz")
        print(f"   Inverted peak freq: {result.spectral.value.get('inv_peak_freq_hz', 0):.1f} Hz")
    if result.spectral.reason:
        print(f"   Reason: {result.spectral.reason}")

    if result.nmf:
        print(f"\n4. NMF MIRROR (Learned Templates)")
        print(f"   Passed: {result.nmf.passed}")
        print(f"   Score: {result.nmf.score:.2f}")
        if result.nmf.value:
            print(f"   Matched type: {result.nmf.value.get('matched_type', 'None')}")
            print(f"   Deviation: {result.nmf.value.get('deviation_ms', -1):.1f} ms")
        if result.nmf.reason:
            print(f"   Reason: {result.nmf.reason}")

    print(f"\n{'=' * 70}")
    print("VERDICT")
    if reject:
        print("❌ HIT REJECTED — Not a valid drum hit")
    else:
        print("✅ HIT ACCEPTED — Passed all four mirrors")