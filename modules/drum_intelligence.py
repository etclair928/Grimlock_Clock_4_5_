#!/usr/bin/env python3
"""
drum_intelligence.py — NMF-First Drum Detection and Classification

Core principles:
1. NMF is PRIMARY — learns drum templates from the audio itself
2. Onset detection is SECONDARY — catches hits NMF might miss
3. Cross-stick has DEDICATED PATH — checked before anything else
4. Band-based PRIORS — soft guidance, not hard rules
5. Decay-based classification — ride vs hi-hat vs crash (4.4 wisdom)
6. Ride pattern validation — tempo feedback from ride cymbal (4.4 wisdom)
7. Integration with Schoenberg Mirror — validation after detection

"No hardcoded thresholds. Let the audio teach us what drums sound like."
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

# Phase 1 & 2 imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from order_types import DrumType, SourceType, DrumHit
from core.audio_utils import (
    ensure_float32, clamp_audio, extract_window
)
from core.fft_helpers import spectral_centroid, zero_crossing_rate

# Import Schoenberg Mirror for validation
from modules.schoenberg_mirror import SchoenbergMirror

# Optional NMF import with fallback
try:
    from sklearn.decomposition import NMF

    NMF_AVAILABLE = True
except ImportError:
    NMF_AVAILABLE = False
    print("⚠️ scikit-learn not available — NMF disabled")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DrumConfig:
    """Configuration for drum intelligence module."""

    # NMF settings
    nmf_components: int = 5
    nmf_max_iter: int = 400
    nmf_activation_threshold: float = 0.12
    nmf_peak_distance: int = 5

    # Onset detection
    onset_backtrack: bool = True
    onset_merge_ms: float = 30.0

    # Cross-stick detection
    cross_stick_threshold: float = 0.55

    # Decay-based classification (from 4.4 wisdom)
    hihat_closed_decay_max: float = 0.12
    hihat_open_decay_max: float = 0.38
    ride_decay_min: float = 0.35

    # Ride pattern validation (from 4.4 wisdom)
    ride_pattern_min_hits: int = 20
    ride_pattern_confidence_threshold: float = 0.7

    # Band priors (soft boosts)
    band_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'sub': {'KICK': 0.25},
        'body': {'SNARE': 0.15, 'TOM': 0.15},
        'presence': {'SNARE': 0.15, 'RIDE': 0.10},
        'air': {'RIDE': 0.20, 'HIHAT_OPEN': 0.15}
    })

    # Confidence thresholds
    min_confidence: float = 0.35
    nmf_override_confidence: float = 0.85

    # Grid snapping
    grid_tolerance_ms: float = 65.0
    grid_penalty: float = 0.5

    # Sample rate
    sample_rate: int = 22050
    hop_length: int = 512


# ============================================================================
# 4.4 WISDOM: DECAY RATIO FOR HI-HAT CLASSIFICATION
# ============================================================================

def get_decay_ratio(audio: np.ndarray, onset_time: float, sr: int) -> float:
    """
    Calculate decay ratio for open/closed hat classification.
    Higher ratio = more open/longer sustain.

    From 4.4 wisdom - essential for distinguishing closed hat, open hat, and ride.
    """
    start = int(onset_time * sr)
    window_samples = int(sr * 0.2)
    end = min(len(audio), start + window_samples)

    if end - start < int(sr * 0.05):
        return 0.0

    segment = audio[start:end]
    envelope = np.abs(segment)

    # Peak in first 20ms
    peak_window = envelope[:int(sr * 0.02)]
    peak = np.max(peak_window) if len(peak_window) > 0 else 0

    if peak < 1e-6:
        return 0.0

    # Energy in last 50ms vs peak
    tail_start = int(sr * 0.15)
    if len(envelope) > tail_start:
        tail_energy = np.mean(envelope[tail_start:])
        decay_ratio = tail_energy / peak
    else:
        decay_ratio = 0.0

    return min(1.0, decay_ratio)


def classify_hihat_by_decay(decay_ratio: float, config: DrumConfig) -> Tuple[str, float]:
    """
    Classify hi-hat as open or closed based on decay ratio.

    From 4.4 wisdom - closed hat decays fast, open hat sustains.

    Args:
        decay_ratio: Ratio of tail energy to peak energy (0-1)
        config: DrumConfig with threshold values

    Returns:
        (hat_type, confidence) where hat_type is 'HIHAT_CLOSED' or 'HIHAT_OPEN'
    """
    if decay_ratio < config.hihat_closed_decay_max:
        return 'HIHAT_CLOSED', 0.85
    elif decay_ratio < config.hihat_open_decay_max:
        return 'HIHAT_OPEN', 0.75
    else:
        # Longer sustain suggests ride, not hi-hat
        return 'RIDE', 0.60


# ============================================================================
# 4.4 WISDOM: RIDE PATTERN VALIDATION & TEMPO FEEDBACK
# ============================================================================

def fix_tempo_by_division(tempo: float) -> float:
    """
    Fix tempo by dividing until it's in a reasonable range.

    From 4.4 wisdom - prevents 240 BPM from being mis-detected as 120 BPM
    when the ride pattern reveals the true tempo.
    """
    while tempo > 200:
        for divisor in [2, 3, 4]:
            candidate = tempo / divisor
            if 40 <= candidate <= 200:
                tempo = candidate
                break
        else:
            tempo /= 2
    return tempo


def validate_ride_pattern(ride_hits: List, config: DrumConfig) -> Tuple[bool, float]:
    """
    Validate that ride hits form a consistent pattern.

    From 4.4 wisdom - prevents tempo overrides from sparse/unreliable hits.

    Args:
        ride_hits: List of DrumHit objects or dicts with 'time' attribute
        config: DrumConfig with threshold values

    Returns:
        (is_valid, confidence)
    """
    if len(ride_hits) < config.ride_pattern_min_hits:
        return False, 0.0

    # Extract times whether they're DrumHit objects or dicts
    if hasattr(ride_hits[0], 'time'):
        ride_times = [h.time for h in ride_hits]
    else:
        ride_times = [h['time'] for h in ride_hits]

    iois = np.diff(ride_times)
    if len(iois) < 2:
        return False, 0.0

    mean_ioi = np.mean(iois)
    if mean_ioi <= 0:
        return False, 0.0

    ioi_std = np.std(iois) / mean_ioi
    confidence = 1.0 - min(1.0, ioi_std)

    return confidence > config.ride_pattern_confidence_threshold, confidence


def detect_ride_pattern_with_confidence(ride_times: List[float], tempo: float) -> Tuple[bool, float, float]:
    """
    Detect ride pattern and optionally correct tempo.

    From 4.4 wisdom - ride cymbal pattern can reveal true tempo in sparse jazz.

    Args:
        ride_times: List of ride hit times in seconds
        tempo: Current estimated tempo in BPM

    Returns:
        (pattern_detected, confidence, corrected_tempo)
    """
    ride_times = np.asarray(ride_times)
    if len(ride_times) < 30:
        return False, 0.0, tempo

    spacings = np.diff(ride_times)
    if len(spacings) == 0:
        return False, 0.0, tempo

    median_spacing = float(np.median(spacings))
    if median_spacing <= 0:
        return False, 0.0, tempo

    beat_duration = 60.0 / tempo if tempo > 0 else 0.5
    triplet_ratio = median_spacing / (beat_duration / 3)

    # Check if ride is playing triplets (common in jazz swing)
    if 0.85 < triplet_ratio < 1.15:
        pattern_strength = len(ride_times) // 3
        if pattern_strength > 10:
            every_third_ioi = np.diff(ride_times[::3])
            if len(every_third_ioi) > 0:
                mean_beat_ioi = float(np.mean(every_third_ioi))
                corrected_tempo = 60.0 / mean_beat_ioi if mean_beat_ioi > 0 else tempo
            else:
                corrected_tempo = tempo
            return True, 0.95, fix_tempo_by_division(corrected_tempo)
        return True, 0.90, tempo

    return False, 0.0, tempo


# ============================================================================
# NMF LABEL MAPPING
# ============================================================================

def map_nmf_label_to_drumtype(nmf_label: str) -> str:
    """
    Map NMF output labels to valid DrumType enum values.
    """
    mapping = {
        'HIHAT': 'HIHAT_OPEN',
        'HIHAT_OPEN': 'HIHAT_OPEN',
        'HIHAT_CLOSED': 'HIHAT_CLOSED',
        'TOM': 'TOM_HIGH',
        'TOM_HIGH': 'TOM_HIGH',
        'TOM_LOW': 'TOM_LOW',
        'KICK': 'KICK',
        'SNARE': 'SNARE',
        'RIDE': 'RIDE',
        'CRASH': 'CRASH',
        'CROSS_STICK': 'CROSS_STICK',
        'OTHER': 'UNKNOWN'
    }
    return mapping.get(nmf_label, 'UNKNOWN')


# ============================================================================
# NMF DRUM DETECTOR (Primary)
# ============================================================================

class NMFDrumDetector:
    """
    NMF-based drum detection — learns templates from the audio itself.

    This is the PRIMARY detector. It finds hits by learning what drums
    sound like in this specific recording, then detecting when those
    learned patterns activate.
    """

    def __init__(self, config: DrumConfig):
        self.config = config
        self._nmf_hits: Dict[str, List[float]] = {}
        self._nmf_components = None
        self._nmf_activations = None
        self._component_labels: Dict[int, str] = {}

    def fit(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        """
        Run NMF decomposition and learn drum templates.

        Args:
            audio: Drum stem audio
            sr: Sample rate

        Returns:
            dict: drum_type -> list of onset times in seconds
        """
        if not NMF_AVAILABLE:
            return {}

        try:
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=64, hop_length=self.config.hop_length,
                fmin=20, fmax=sr // 2
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

            # Run NMF with KL divergence (better for sparse drum patterns)
            model = NMF(n_components=self.config.nmf_components,
                        beta_loss='kullback-leibler',
                        solver='mu',
                        max_iter=self.config.nmf_max_iter,
                        random_state=42)

            W_T = model.fit_transform(S_norm.T)
            H = W_T.T  # activations (n_components, n_frames)
            W = model.components_.T  # templates (n_mels, n_components)

            self._nmf_components = W
            self._nmf_activations = H

            # Auto-label components by spectral centroid
            self._component_labels = self._auto_label_components(W, sr)

            # Extract hit times from activations
            frame_times = librosa.frames_to_time(
                np.arange(H.shape[1]), sr=sr, hop_length=self.config.hop_length
            )

            self._nmf_hits = {}
            for comp_idx, drum_type in self._component_labels.items():
                activation = H[comp_idx, :].copy()
                if np.max(activation) > 0:
                    activation /= np.max(activation)

                peaks, _ = find_peaks(activation,
                                      height=self.config.nmf_activation_threshold,
                                      distance=self.config.nmf_peak_distance)
                if len(peaks) > 0:
                    mapped_type = map_nmf_label_to_drumtype(drum_type)
                    if mapped_type != 'UNKNOWN':
                        new_times = frame_times[peaks].tolist()
                        if mapped_type in self._nmf_hits:
                            # FIX: merge instead of overwrite.
                            # When two NMF components map to the same drum type
                            # (e.g. two snare templates) the original code silently
                            # discarded the first one.  Now we merge and de-duplicate
                            # within a 20 ms window so no hit is lost.
                            existing = self._nmf_hits[mapped_type]
                            combined = sorted(existing + new_times)
                            deduped  = []
                            for t in combined:
                                if not deduped or (t - deduped[-1]) > 0.020:
                                    deduped.append(t)
                            self._nmf_hits[mapped_type] = deduped
                        else:
                            self._nmf_hits[mapped_type] = new_times

            total_hits = sum(len(v) for v in self._nmf_hits.values())
            if total_hits > 0:
                print(f"🥁 NMF: {total_hits} hits across {list(self._nmf_hits.keys())}")

            return self._nmf_hits

        except Exception as e:
            print(f"⚠️ NMF detector failed: {e}")
            return {}

    def _auto_label_components(self, W: np.ndarray, sr: int) -> Dict[int, str]:
        """
        Auto-label NMF components by spectral centroid (relative ordering).
        Maps to valid DrumType enum values.
        """
        # Get mel frequencies for centroid calculation
        mel_freqs = librosa.mel_frequencies(n_mels=W.shape[0], fmin=20, fmax=sr // 2)

        # Calculate centroid for each component
        centroids = []
        for i in range(W.shape[1]):
            template = W[:, i]
            if np.sum(template) > 0:
                centroid = np.sum(mel_freqs * template) / (np.sum(template) + 1e-8)
            else:
                centroid = 0
            centroids.append((i, centroid))

        # Sort by centroid (lowest to highest)
        centroids.sort(key=lambda x: x[1])
        n = len(centroids)

        # Relative labeling with VALID enum values
        labels = {}

        if n > 0:
            labels[centroids[0][0]] = 'KICK'  # lowest centroid
        if n > 1:
            labels[centroids[1][0]] = 'SNARE'  # second lowest
        if n > 2:
            labels[centroids[n // 2][0]] = 'TOM'  # middle (generic tom)
        if n > 3:
            labels[centroids[-2][0]] = 'RIDE'  # second highest
        if n > 4:
            labels[centroids[-1][0]] = 'HIHAT'  # highest centroid

        # Fill remaining as OTHER
        for i, _ in centroids:
            if i not in labels:
                labels[i] = 'OTHER'

        return labels

    def get_hits(self) -> Dict[str, List[float]]:
        """Return detected NMF hits."""
        return self._nmf_hits

    def get_type_at_time(self, time: float, tolerance_ms: float = 55.0) -> Optional[str]:
        """Get NMF-classified type nearest to time."""
        tolerance = tolerance_ms / 1000.0
        best_type = None
        best_dist = float('inf')

        for drum_type, times in self._nmf_hits.items():
            for t in times:
                d = abs(t - time)
                if d < tolerance and d < best_dist:
                    best_dist = d
                    best_type = drum_type

        return best_type


# ============================================================================
# ONSET DETECTOR (Secondary)
# ============================================================================

class OnsetDrumDetector:
    """
    Onset-based drum detection — secondary detector.

    Catches hits that NMF might miss (ghost notes, soft cymbals).
    Also provides hit candidates for feature extraction.
    """

    def __init__(self, config: DrumConfig):
        self.config = config

    def detect(self, audio: np.ndarray, sr: int) -> List[float]:
        """
        Detect onsets in audio.

        Args:
            audio: Drum stem audio
            sr: Sample rate

        Returns:
            list of onset times in seconds
        """
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=self.config.onset_backtrack
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        return onset_times.tolist()

    def detect_with_strength(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect onsets with onset strength values.

        Returns:
            list of dicts with 'time' and 'strength' keys
        """
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=self.config.onset_backtrack,
            units='frames'
        )
        onset_strength = librosa.onset.onset_strength(y=audio, sr=sr)

        hits = []
        for frame in onset_frames:
            if frame < len(onset_strength):
                strength = float(onset_strength[frame])
                time = librosa.frames_to_time(frame, sr=sr)
                hits.append({'time': time, 'strength': strength, 'source': 'onset'})

        return hits


# ============================================================================
# CROSS-STICK DETECTOR (Dedicated Path)
# ============================================================================

class CrossStickDetector:
    """
    Dedicated cross-stick (rim click) detector.

    Cross-stick has a unique spectral signature: woody knock in 800-1200 Hz,
    very fast attack, very short decay, minimal low-end energy.

    This runs BEFORE other classification to ensure cross-stick is never
    mislabeled as snare.
    """

    def __init__(self, config: DrumConfig):
        self.config = config

    def detect(self, audio: np.ndarray, onset_time: float, sr: int) -> Tuple[bool, float]:
        """
        Detect if a hit is a cross-stick.

        Args:
            audio: Full audio array
            onset_time: Time of the hit in seconds
            sr: Sample rate

        Returns:
            (is_cross_stick, confidence)
        """
        start = max(0, int(onset_time * sr))
        end = min(len(audio), start + int(sr * 0.15))

        if end - start < 32:
            return False, 0.0

        segment = audio[start:end] * np.hanning(end - start)
        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(end - start, 1 / sr)
        total = np.sum(spectrum) + 1e-10

        # Feature 1: Energy in 800-1200 Hz (the "wood knock")
        cs_energy = np.sum(spectrum[(freqs >= 800) & (freqs <= 1200)]) / total

        # Feature 2: Low-end energy (should be minimal)
        low_energy = np.sum(spectrum[(freqs >= 30) & (freqs <= 100)]) / total

        # Feature 3: Attack time (should be very fast)
        envelope = np.abs(audio[start:end])
        peak_idx = np.argmax(envelope[:int(sr * 0.02)])
        attack_ms = (peak_idx / sr) * 1000 if peak_idx > 0 else 10.0

        # Feature 4: Decay time (should be very short)
        if peak_idx + int(sr * 0.02) < len(envelope):
            tail = envelope[peak_idx + int(sr * 0.02):]
            decay_ms = (len(tail) / sr) * 1000 if len(tail) > 0 else 100
        else:
            decay_ms = 100

        # Score calculation
        score = 0.0
        if cs_energy > 0.10:
            score += 0.35
        if low_energy < 0.10:
            score += 0.25
        if attack_ms < 4.0:
            score += 0.20
        if decay_ms < 120:
            score += 0.20

        is_cross_stick = score > self.config.cross_stick_threshold

        return is_cross_stick, min(1.0, score)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_drum_features(audio: np.ndarray, onset_time: float, sr: int) -> Dict[str, float]:
    """
    Extract features for drum hit classification.

    Returns:
        dict with features: centroid, zcr, energy, decay_ratio, attack_ms
    """
    features = {}

    # Extract window around onset (80ms)
    start = max(0, int(onset_time * sr))
    end = min(len(audio), start + int(0.08 * sr))

    if end - start < 32:
        return features

    window = audio[start:end]

    # Spectral centroid
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1 / sr)
    total = np.sum(spectrum) + 1e-10
    features['centroid'] = np.sum(freqs * spectrum) / total

    # Energy
    features['energy'] = float(np.sum(window ** 2))

    # Zero-crossing rate
    features['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(window)))

    # Fast decay estimate (tail/peak ratio)
    peak = np.max(np.abs(window)) + 1e-6
    tail_start = min(len(window), int(0.02 * sr))
    tail = np.mean(np.abs(window[tail_start:])) if tail_start < len(window) else 0
    features['decay_ratio'] = tail / peak

    # Attack time
    envelope = np.abs(window)
    peak_idx = np.argmax(envelope[:int(sr * 0.02)])
    features['attack_ms'] = (peak_idx / sr) * 1000 if peak_idx > 0 else 10.0

    return features


# ============================================================================
# DRUM CLASSIFIER (Merges NMF + Onset + Cross-Stick + 4.4 Wisdom)
# ============================================================================

class DrumClassifier:
    """
    Main drum classifier that merges all detection paths.

    Priority order:
    1. Cross-stick detector (dedicated path)
    2. NMF detector (primary)
    3. Onset detector (secondary, with feature-based classification)
    4. 4.4 Wisdom: Decay ratio for hi-hat classification
    5. 4.4 Wisdom: Ride pattern validation

    Also integrates with Schoenberg Mirror for validation.
    """

    def __init__(self, config: DrumConfig):
        self.config = config
        self.nmf_detector = NMFDrumDetector(config)
        self.onset_detector = OnsetDrumDetector(config)
        self.cross_stick = CrossStickDetector(config)

    def classify(self, audio: np.ndarray, sr: int,
                 nmf_hits: Optional[Dict[str, List[float]]] = None,
                 tempo: float = 120.0) -> List[DrumHit]:
        """
        Classify drum hits from audio.

        Args:
            audio:     Drum stem audio
            sr:        Sample rate
            nmf_hits:  Pre-computed NMF hits (optional, will compute if not provided)
            tempo:     Current tempo in BPM — used for tempo-aware timing windows
                       (e.g. at 60 BPM a 65 ms window is proportionally tighter
                       than at 200 BPM; we scale the merge window accordingly)

        Returns:
            list of DrumHit objects
        """
        audio = ensure_float32(audio)
        audio = clamp_audio(audio)

        if nmf_hits is None:
            nmf_hits = self.nmf_detector.fit(audio, sr)

        onset_hits = self.onset_detector.detect_with_strength(audio, sr)

        # Tempo-aware merge window: at 120 BPM a 30 ms window is one 64th note.
        # Scale proportionally so fast tempos don't merge unrelated hits.
        beat_duration_ms   = (60.0 / max(tempo, 40.0)) * 1000.0
        merge_window_ms    = max(15.0, min(40.0, beat_duration_ms / 8.0))
        merge_window       = merge_window_ms / 1000.0

        all_candidates = []
        for drum_type, times in nmf_hits.items():
            for t in times:
                all_candidates.append({
                    'time': t, 'strength': 0.9,
                    'source': 'nmf', 'nmf_type': drum_type
                })
        for hit in onset_hits:
            all_candidates.append({
                'time': hit['time'], 'strength': hit['strength'],
                'source': 'onset', 'nmf_type': None
            })

        all_candidates.sort(key=lambda x: x['time'])
        merged_candidates = []
        for cand in all_candidates:
            if (not merged_candidates or
                    abs(cand['time'] - merged_candidates[-1]['time']) >= merge_window):
                merged_candidates.append(cand)
            elif cand['strength'] > merged_candidates[-1]['strength']:
                merged_candidates[-1] = cand

        drum_hits = []

        for cand in merged_candidates:
            # Step 1: Cross-stick (dedicated path — always checked first)
            is_cs, cs_conf = self.cross_stick.detect(audio, cand['time'], sr)
            if is_cs:
                # Confidence gating: cross-stick must clear min_confidence
                if cs_conf >= self.config.min_confidence:
                    velocity = int(np.clip(cand['strength'] * 80, 20, 100))
                    drum_hits.append(DrumHit(
                        time=cand['time'], drum_type=DrumType.CROSS_STICK,
                        confidence=cs_conf, velocity=velocity,
                        source=SourceType.CROSS_STICK
                    ))
                continue

            # Step 2: NMF override (high-confidence primary path)
            if cand.get('nmf_type'):
                mapped_type = map_nmf_label_to_drumtype(cand['nmf_type'])
                try:
                    drum_type = DrumType(mapped_type)
                    # Velocity normalisation per drum type
                    # Kick: loud (80-127), snare: mid (60-110), cymbals: soft (40-90)
                    vel_ranges = {
                        DrumType.KICK:         (80, 127),
                        DrumType.SNARE:        (60, 110),
                        DrumType.CROSS_STICK:  (40,  80),
                        DrumType.HIHAT_CLOSED: (40,  90),
                        DrumType.HIHAT_OPEN:   (45,  95),
                        DrumType.RIDE:         (40,  85),
                        DrumType.CRASH:        (70, 115),
                        DrumType.TOM_HIGH:     (55, 105),
                        DrumType.TOM_LOW:      (60, 110),
                    }
                    lo, hi = vel_ranges.get(drum_type, (50, 100))
                    velocity = int(np.clip(cand['strength'] * (hi - lo) + lo, lo, hi))
                    conf     = self.config.nmf_override_confidence
                    # Confidence gating
                    if conf >= self.config.min_confidence:
                        drum_hits.append(DrumHit(
                            time=cand['time'], drum_type=drum_type,
                            confidence=conf, velocity=velocity,
                            source=SourceType.NMF
                        ))
                except ValueError:
                    pass
                continue

            # Step 3: Feature-based classification (onset fallback + 4.4 wisdom)
            features = extract_drum_features(audio, cand['time'], sr)
            if not features:
                continue

            drum_type, confidence = self._classify_by_features(
                features, audio, cand['time'], sr)

            # Confidence gating — only emit if above minimum
            if confidence >= self.config.min_confidence:
                lo, hi = {
                    DrumType.KICK:         (75, 127),
                    DrumType.SNARE:        (55, 105),
                    DrumType.HIHAT_CLOSED: (35,  85),
                    DrumType.HIHAT_OPEN:   (40,  90),
                    DrumType.RIDE:         (35,  80),
                }.get(drum_type, (45, 95))
                # Use energy-based velocity clamped to type range
                raw_energy = features.get('energy', 0.5)
                norm_energy = float(np.clip(raw_energy / (raw_energy + 1e-4), 0, 1))
                velocity = int(np.clip(norm_energy * (hi - lo) + lo, lo, hi))
                drum_hits.append(DrumHit(
                    time=cand['time'], drum_type=drum_type,
                    confidence=confidence, velocity=velocity,
                    source=SourceType.ONSET
                ))

        drum_hits.sort(key=lambda x: x.time)
        return drum_hits

    def _classify_by_features(self, features: Dict[str, float],
                              audio: np.ndarray = None,
                              onset_time: float = None,
                              sr: int = None) -> Tuple[DrumType, float]:
        """
        Classify drum type using extracted features.

        Updated with 4.4 wisdom for hi-hat decay classification and ride pattern.
        """
        centroid = features.get('centroid', 1000)
        decay = features.get('decay_ratio', 0.3)
        zcr = features.get('zcr', 0.1)
        attack_ms = features.get('attack_ms', 10)

        # KICK: low centroid, low ZCR, moderate decay
        if centroid < 120:
            return DrumType.KICK, 0.7

        # SNARE: mid centroid, mid-high ZCR, moderate decay
        if 150 < centroid < 400:
            return DrumType.SNARE, 0.6

        # HI-HAT: high ZCR, decay-based open/closed (4.4 wisdom)
        if zcr > 0.2:
            hat_type, hat_conf = classify_hihat_by_decay(decay, self.config)
            if hat_type == 'HIHAT_CLOSED':
                return DrumType.HIHAT_CLOSED, hat_conf
            elif hat_type == 'HIHAT_OPEN':
                return DrumType.HIHAT_OPEN, hat_conf
            else:
                return DrumType.RIDE, hat_conf

        # RIDE: mid-high centroid, longer decay (4.4 wisdom)
        if centroid > 800 and decay > self.config.ride_decay_min:
            return DrumType.RIDE, 0.65

        # TOM: mid centroid, moderate decay
        if 200 < centroid < 600:
            return DrumType.TOM_HIGH, 0.55

        return DrumType.UNKNOWN, 0.3


# ============================================================================
# MAIN DRUM INTELLIGENCE MODULE
# ============================================================================

class DrumIntelligence:
    """
    Main drum intelligence module.

    Orchestrates:
    - NMF detection (primary)
    - Onset detection (secondary)
    - Cross-stick detection (dedicated path)
    - Feature-based classification
    - 4.4 Wisdom: Decay ratio hi-hat classification
    - 4.4 Wisdom: Ride pattern validation & tempo feedback
    - Integration with Schoenberg Mirror
    - Grid snapping
    """

    def __init__(self, config: Optional[DrumConfig] = None):
        self.config = config or DrumConfig()
        self.classifier = DrumClassifier(self.config)
        self.nmf_detector = self.classifier.nmf_detector
        self.schoenberg = None  # Will be set if provided

    def set_schoenberg_mirror(self, mirror: SchoenbergMirror):
        """Attach Schoenberg Mirror for validation."""
        self.schoenberg = mirror

    def validate_ride_pattern(self, drum_hits: List[DrumHit]) -> Tuple[bool, float]:
        """
        Validate ride pattern consistency.

        From 4.4 wisdom - ensures ride hits form a reliable pattern
        before using them for tempo feedback.
        """
        ride_hits = [h for h in drum_hits if h.drum_type in (DrumType.RIDE, DrumType.HIHAT_OPEN, DrumType.HIHAT_CLOSED)]
        return validate_ride_pattern(ride_hits, self.config)

    def get_ride_tempo_correction(self, drum_hits: List[DrumHit], current_tempo: float) -> Tuple[float, bool, float]:
        """
        Get tempo correction from ride pattern.

        From 4.4 wisdom - ride cymbal pattern can reveal true tempo.

        Returns:
            (corrected_tempo, pattern_detected, confidence)
        """
        ride_hits = [h for h in drum_hits if h.drum_type == DrumType.RIDE]
        ride_times = [h.time for h in ride_hits]

        pattern_detected, confidence, corrected_tempo = detect_ride_pattern_with_confidence(
            ride_times, current_tempo
        )

        if pattern_detected and confidence > self.config.ride_pattern_confidence_threshold:
            return corrected_tempo, True, confidence

        return current_tempo, False, 0.0

    def process(self, audio: np.ndarray, sr: int,
                use_validation: bool = True,
                state=None,           # Optional[StateManager]
                consensus_tempo: float = 120.0) -> Tuple[List[DrumHit], Dict]:
        """
        Process drum audio and return classified hits.

        Args:
            audio:            Drum stem audio
            sr:               Sample rate
            use_validation:   Apply Schoenberg Mirror validation
            state:            Optional StateManager — reads consensus tempo,
                              writes instrument / swing / phrase evidence
            consensus_tempo:  Tempo from RhythmEngine (used for tempo-aware
                              timing windows; also read from state if available)

        Returns:
            (drum_hits, metadata)
        """
        # Read consensus tempo from StateManager if available
        if state:
            try:
                tempo_from_state = state._context.tempo
                if tempo_from_state > 0:
                    consensus_tempo = tempo_from_state
                    print(f"🥁 DrumIntel: Using consensus tempo {consensus_tempo:.1f} BPM from state")
            except Exception:
                pass

        # Run NMF first (learns templates)
        nmf_hits = self.nmf_detector.fit(audio, sr)

        # Classify hits — pass tempo for tempo-aware timing windows
        drum_hits = self.classifier.classify(audio, sr, nmf_hits, tempo=consensus_tempo)

        # Apply Schoenberg Mirror validation if available
        if use_validation and self.schoenberg:
            validated_hits = []
            for hit in drum_hits:
                result = self.schoenberg.validate_hit(audio, hit.time, use_nmf=True)
                reject, reason = self.schoenberg.should_reject(result)
                if reject:
                    continue
                multiplier = self.schoenberg.get_confidence_multiplier(result)
                hit.confidence = min(1.0, hit.confidence * multiplier)
                # Confidence gate again after multiplier (may have dropped below min)
                if hit.confidence >= self.config.min_confidence:
                    validated_hits.append(hit)
            drum_hits = validated_hits

        # Write discoveries to StateManager
        if state:
            try:
                duration = len(audio) / sr
                # Instrument detections
                type_set = {h.drum_type.value for h in drum_hits}
                for inst in type_set:
                    state.add_evidence("instrument", inst.lower(), 0.8,
                                       "drum_intel", start_time=0.0, end_time=duration)

                # Swing evidence from ride cymbal IOI
                ride_hits = [h for h in drum_hits if h.drum_type == DrumType.RIDE]
                if len(ride_hits) >= 8:
                    ride_times = np.array([h.time for h in ride_hits])
                    iois       = np.diff(ride_times)
                    beat_dur   = 60.0 / max(consensus_tempo, 1.0)
                    third_dur  = beat_dur / 3.0
                    near_third = iois[np.abs(iois - third_dur) < third_dur * 0.2]
                    if len(near_third) > 4:
                        swing_ratio = float(np.clip(np.mean(near_third) / third_dur, 0.5, 2.0))
                        state.add_evidence("swing", swing_ratio, 0.75,
                                           "drum_intel", start_time=0.0, end_time=duration)

                # Mark low-confidence segments as problem areas
                for hit in drum_hits:
                    if hit.confidence < 0.4:
                        state.add_evidence("problem_segment", hit.time, hit.confidence,
                                           "drum_intel",
                                           start_time=hit.time - 0.2,
                                           end_time=hit.time + 0.2)
            except Exception as e:
                print(f"   ⚠️ StateManager write failed (non-fatal): {e}")

        # Collect metadata
        ride_hits     = [h for h in drum_hits if h.drum_type == DrumType.RIDE]
        hihat_closed  = [h for h in drum_hits if h.drum_type == DrumType.HIHAT_CLOSED]
        hihat_open    = [h for h in drum_hits if h.drum_type == DrumType.HIHAT_OPEN]

        metadata = {
            'total_hits':        len(drum_hits),
            'nmf_hits':          sum(len(v) for v in nmf_hits.values()),
            'consensus_tempo':   consensus_tempo,
            'type_counts':       {dt.value: len([h for h in drum_hits if h.drum_type == dt])
                                  for dt in DrumType},
            'ride_hits':         len(ride_hits),
            'hihat_closed':      len(hihat_closed),
            'hihat_open':        len(hihat_open),
            'ride_pattern_valid': False,
            'ride_tempo_correction': None,
        }

        if len(ride_hits) >= self.config.ride_pattern_min_hits:
            is_valid, confidence = self.validate_ride_pattern(drum_hits)
            metadata['ride_pattern_valid']       = is_valid
            metadata['ride_pattern_confidence']  = confidence

        return drum_hits, metadata


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Test Drum Intelligence Module")
    parser.add_argument("audio_file", help="Path to drum audio file")
    args = parser.parse_args()

    # Load audio
    audio, sr = librosa.load(args.audio_file, sr=22050)

    # Create drum intelligence module
    drum_intel = DrumIntelligence()

    # Process
    hits, metadata = drum_intel.process(audio, sr, use_validation=False)

    print(f"\n{'=' * 60}")
    print(f"Drum Intelligence Test (with 4.4 Wisdom)")
    print(f"{'=' * 60}")
    print(f"Total hits: {len(hits)}")
    print(f"NMF hits: {metadata['nmf_hits']}")

    print(f"\nDrum type counts:")
    for dt, count in metadata['type_counts'].items():
        if count > 0:
            print(f"  {dt}: {count}")

    print(f"\n4.4 Wisdom Stats:")
    print(f"  Ride hits: {metadata['ride_hits']}")
    print(f"  Hi-hat closed: {metadata['hihat_closed']}")
    print(f"  Hi-hat open: {metadata['hihat_open']}")
    print(f"  Ride pattern valid: {metadata.get('ride_pattern_valid', False)}")

    print(f"\nFirst 15 hits:")
    for hit in hits[:15]:
        print(f"  {hit.time:.3f}s - {hit.drum_type.value} (conf={hit.confidence:.2f})")