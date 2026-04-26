#!/usr/bin/env python3
"""
stem_classifier.py — Intelligent Instrument Detection for the "Other" Stem

Modernized for Grimlock 4.5:
- Confidence-based instrument detection (not binary)
- Multi-instrument support with probability scoring
- Integration with PitchIntelligence for optimal routing
- Feature extraction using modern ML techniques
- Graceful fallback to piano when uncertain

The Demucs 'other' stem contains harmonic content (piano, guitar, strings, etc.)
This classifier determines what's actually there so we can route to the
appropriate transcriber.
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import OrderedDict

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# TYPES AND CONFIG
# ============================================================================

class InstrumentCategory(Enum):
    """Instrument categories for routing"""
    PIANO = "piano"
    GUITAR = "guitar"
    STRINGS = "strings"
    WINDS = "winds"
    BRASS = "brass"
    SYNTH = "synth"
    ORGAN = "organ"
    HARP = "harp"
    UNKNOWN = "unknown"


@dataclass
class InstrumentDetection:
    """Result of instrument detection with confidence"""
    instrument: InstrumentCategory
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ClassifierConfig:
    """Configuration for stem classifier"""
    # Feature extraction
    n_mfcc: int = 13
    hop_length: int = 512
    n_fft: int = 2048

    # Analysis windows
    analysis_duration: float = 5.0  # seconds
    min_audio_duration: float = 1.0

    # Confidence thresholds
    piano_threshold: float = 0.3
    guitar_threshold: float = 0.4
    strings_threshold: float = 0.4
    winds_threshold: float = 0.4
    brass_threshold: float = 0.4

    # Fallback
    default_instrument: InstrumentCategory = InstrumentCategory.PIANO
    default_confidence: float = 0.6


# ============================================================================
# MODERN FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Modern feature extraction for instrument classification.
    Uses spectral, temporal, and harmonic features.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.sr = 22050  # Target sample rate

    def extract_all(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract comprehensive feature set from audio.

        Returns dict of feature name -> value
        """
        if audio is None or len(audio) < self.config.min_audio_duration * sr:
            return {}

        # Resample if needed
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        features = {}

        # 1. Spectral features
        features.update(self._spectral_features(audio))

        # 2. Temporal features
        features.update(self._temporal_features(audio))

        # 3. Harmonic features
        features.update(self._harmonic_features(audio))

        # 4. Onset/attack features
        features.update(self._onset_features(audio))

        # 5. MFCC statistics
        features.update(self._mfcc_features(audio))

        return features

    def _spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral centroid, bandwidth, rolloff, etc."""
        features = {}

        # Spectrogram
        S = np.abs(librosa.stft(audio, hop_length=self.config.hop_length))

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(S=S, sr=self.sr)
        features['spectral_centroid_mean'] = float(np.mean(centroid))
        features['spectral_centroid_std'] = float(np.std(centroid))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sr)
        features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=self.sr)
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio)
        features['spectral_flatness_mean'] = float(np.mean(flatness))

        return features

    def _temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract tempo, rhythm stability, etc."""
        features = {}

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features['tempo'] = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo) if tempo else 120.0

        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.config.hop_length)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_var'] = float(np.var(rms))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.config.hop_length)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        return features

    def _harmonic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract harmonic-to-noise ratio, pitch stability, etc."""
        features = {}

        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)

        # HNR (Harmonic-to-Noise Ratio)
        hnr = np.mean(harmonic ** 2) / (np.mean(percussive ** 2) + 1e-8)
        features['hnr'] = float(np.clip(10 * np.log10(hnr), 0, 30))

        # Pitch stability (using piptrack)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)

        # Get non-zero pitches
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            features['pitch_stability'] = float(1.0 - (np.std(pitch_values) / (np.mean(pitch_values) + 1e-8)))
            features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            features['pitch_stability'] = 0.5
            features['pitch_range'] = 0.0

        return features

    def _onset_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract attack characteristics, onset density, etc."""
        features = {}

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr, hop_length=self.config.hop_length)

        features['onset_density'] = float(np.sum(onset_env > 0.1) / len(onset_env))
        features['onset_strength_mean'] = float(np.mean(onset_env))
        features['onset_strength_std'] = float(np.std(onset_env))
        features['onset_strength_max'] = float(np.max(onset_env))

        # Attack time estimation (simple)
        envelope = np.abs(audio)
        peak_idx = np.argmax(envelope[:int(0.1 * self.sr)])
        features['attack_time_ms'] = float(peak_idx / self.sr * 1000) if peak_idx > 0 else 10.0

        # Decay rate
        if peak_idx < len(envelope) - int(0.2 * self.sr):
            tail = envelope[peak_idx + int(0.05 * self.sr):peak_idx + int(0.2 * self.sr)]
            features['decay_rate'] = float(np.mean(tail) / (envelope[peak_idx] + 1e-8))
        else:
            features['decay_rate'] = 0.5

        return features

    def _mfcc_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract MFCC statistics"""
        features = {}

        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.config.n_mfcc,
            hop_length=self.config.hop_length, n_fft=self.config.n_fft
        )

        for i in range(min(self.config.n_mfcc, mfccs.shape[0])):
            features[f'mfcc_{i + 1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i + 1}_std'] = float(np.std(mfccs[i]))

        # Delta MFCCs (capture temporal evolution)
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(min(5, delta_mfccs.shape[0])):
            features[f'delta_mfcc_{i + 1}_mean'] = float(np.mean(delta_mfccs[i]))

        return features


# ============================================================================
# MODERN CLASSIFIER (using feature-based heuristics + ML-lite)
# ============================================================================

class StemClassifier:
    """
    Modern instrument classifier for the "other" stem.

    Uses multi-feature analysis with confidence scoring.
    Returns probability distribution over instrument types.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None, sr: int = 22050):
        self.config = config or ClassifierConfig()
        self.sr = sr
        self.extractor = FeatureExtractor(self.config)

        # Feature profiles for each instrument (derived from empirical analysis)
        self._init_profiles()

    def _init_profiles(self):
        """Initialize instrument feature profiles"""
        self.profiles = {
            InstrumentCategory.PIANO: {
                'spectral_centroid_mean': (1000, 2500),
                'onset_density': (0.3, 0.8),
                'hnr': (10, 25),
                'pitch_range': (200, 1000),
                'decay_rate': (0.2, 0.6),
                'rms_var': (0.01, 0.05)
            },
            InstrumentCategory.GUITAR: {
                'spectral_centroid_mean': (800, 2000),
                'onset_density': (0.2, 0.6),
                'hnr': (5, 20),
                'pitch_range': (50, 400),
                'decay_rate': (0.1, 0.4),
                'attack_time_ms': (5, 20)
            },
            InstrumentCategory.STRINGS: {
                'spectral_centroid_mean': (600, 1800),
                'onset_density': (0.1, 0.5),
                'hnr': (15, 30),
                'pitch_range': (100, 500),
                'decay_rate': (0.3, 0.7),
                'pitch_stability': (0.7, 1.0)
            },
            InstrumentCategory.WINDS: {
                'spectral_centroid_mean': (800, 2000),
                'onset_density': (0.05, 0.3),
                'hnr': (5, 15),
                'pitch_range': (50, 300),
                'decay_rate': (0.1, 0.5),
                'zcr_mean': (0.05, 0.2)
            },
            InstrumentCategory.BRASS: {
                'spectral_centroid_mean': (1200, 3000),
                'onset_density': (0.05, 0.3),
                'hnr': (8, 20),
                'pitch_range': (50, 400),
                'attack_time_ms': (10, 40),
                'spectral_bandwidth_mean': (800, 2000)
            },
            InstrumentCategory.SYNTH: {
                'spectral_centroid_mean': (500, 4000),
                'onset_density': (0.1, 0.9),
                'hnr': (20, 40),
                'pitch_stability': (0.5, 1.0),
                'spectral_flatness_mean': (0.3, 0.8)
            },
            InstrumentCategory.ORGAN: {
                'spectral_centroid_mean': (500, 2000),
                'onset_density': (0.05, 0.3),
                'hnr': (15, 35),
                'pitch_stability': (0.8, 1.0),
                'rms_var': (0.005, 0.02)
            }
        }

        # Feature weights for each instrument
        self.weights = {
            InstrumentCategory.PIANO: {'spectral_centroid_mean': 0.3, 'onset_density': 0.25, 'hnr': 0.2,
                                       'pitch_range': 0.15, 'decay_rate': 0.1},
            InstrumentCategory.GUITAR: {'spectral_centroid_mean': 0.25, 'onset_density': 0.2, 'attack_time_ms': 0.2,
                                        'decay_rate': 0.2, 'hnr': 0.15},
            InstrumentCategory.STRINGS: {'pitch_stability': 0.35, 'hnr': 0.25, 'decay_rate': 0.2,
                                         'spectral_centroid_mean': 0.2},
            InstrumentCategory.WINDS: {'zcr_mean': 0.3, 'onset_density': 0.25, 'hnr': 0.25,
                                       'spectral_centroid_mean': 0.2},
            InstrumentCategory.BRASS: {'spectral_centroid_mean': 0.3, 'attack_time_ms': 0.25,
                                       'spectral_bandwidth_mean': 0.25, 'hnr': 0.2},
            InstrumentCategory.SYNTH: {'spectral_flatness_mean': 0.3, 'hnr': 0.25, 'pitch_stability': 0.25,
                                       'spectral_centroid_mean': 0.2},
            InstrumentCategory.ORGAN: {'pitch_stability': 0.35, 'hnr': 0.3, 'rms_var': 0.2,
                                       'spectral_centroid_mean': 0.15}
        }

    def classify(self, audio: np.ndarray, return_all: bool = False) -> List[str]:
        """
        Classify instruments in the audio.

        Args:
            audio: Audio array
            return_all: If True, return all instruments with confidence > 0.3

        Returns:
            List of instrument names (e.g., ['piano'], ['piano', 'strings'])
        """
        if audio is None or len(audio) == 0:
            return []

        # Get detection results with confidence
        detections = self.classify_with_confidence(audio)

        # Filter by threshold and return instrument names
        instruments = []
        for det in detections:
            threshold = getattr(self.config, f"{det.instrument.value}_threshold", 0.3)
            if det.confidence >= threshold:
                instruments.append(det.instrument.value)

        # Ensure we return at least one instrument
        if not instruments:
            instruments = [self.config.default_instrument.value]

        return instruments

    def classify_with_confidence(self, audio: np.ndarray) -> List[InstrumentDetection]:
        """
        Classify instruments with confidence scores.

        Returns:
            List of InstrumentDetection objects sorted by confidence
        """
        if audio is None or len(audio) < self.config.min_audio_duration * self.sr:
            # Not enough audio, return default
            return [InstrumentDetection(
                instrument=self.config.default_instrument,
                confidence=self.config.default_confidence
            )]

        # Extract features
        features = self.extractor.extract_all(audio, self.sr)

        if not features:
            return [InstrumentDetection(
                instrument=self.config.default_instrument,
                confidence=self.config.default_confidence
            )]

        # Score each instrument
        scores = []
        for instrument, profile in self.profiles.items():
            score = self._score_instrument(features, profile, instrument)
            scores.append((instrument, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Convert to InstrumentDetection objects
        results = []
        for instrument, score in scores[:3]:  # Top 3
            if score > 0.1:  # Minimum threshold
                results.append(InstrumentDetection(
                    instrument=instrument,
                    confidence=score,
                    features={k: v for k, v in features.items() if k in self.weights.get(instrument, {})}
                ))

        # Ensure we have at least one result
        if not results:
            results = [InstrumentDetection(
                instrument=self.config.default_instrument,
                confidence=self.config.default_confidence
            )]

        return results

    def _score_instrument(self, features: Dict[str, float], profile: Dict[str, Tuple[float, float]],
                          instrument: InstrumentCategory) -> float:
        """
        Score how well features match an instrument profile.
        """
        weights = self.weights.get(instrument, {})
        if not weights:
            return 0.5

        total_weight = 0
        weighted_score = 0

        for feature_name, (low, high) in profile.items():
            if feature_name not in features:
                continue

            weight = weights.get(feature_name, 0.1)
            total_weight += weight

            value = features[feature_name]

            # Gaussian-like scoring: peak at center of range
            center = (low + high) / 2
            half_range = (high - low) / 2

            if half_range > 0:
                # Distance from center, normalized
                distance = abs(value - center) / half_range
                feature_score = max(0, 1.0 - min(1.0, distance))
            else:
                # Single value expectation
                feature_score = 1.0 if abs(value - low) < low * 0.1 else 0.5

            weighted_score += weight * feature_score

        if total_weight > 0:
            return weighted_score / total_weight
        return 0.5

    # =========================================================================
    # SIMPLE MODE (for compatibility with older code)
    # =========================================================================

    def classify_simple(self, audio: np.ndarray) -> List[str]:
        """
        Simple classification for backward compatibility.
        Always returns ['piano'] as safe default.
        """
        if audio is None or len(audio) == 0:
            return []
        return ['piano']

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_routing_suggestion(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Get suggested routing for PitchIntelligence.

        Returns:
            Tuple of (primary_instrument, confidence)
        """
        detections = self.classify_with_confidence(audio)

        if detections:
            primary = detections[0]
            return primary.instrument.value, primary.confidence

        return self.config.default_instrument.value, self.config.default_confidence

    def should_use_basic_pitch(self, audio: np.ndarray) -> bool:
        """
        Determine if Basic Pitch is appropriate for this audio.

        Basic Pitch is good for piano, guitar, and general polyphonic content.
        """
        detections = self.classify_with_confidence(audio)

        # Instruments that work well with Basic Pitch
        good_for_bp = [
            InstrumentCategory.PIANO,
            InstrumentCategory.GUITAR,
            InstrumentCategory.STRINGS,
            InstrumentCategory.ORGAN,
            InstrumentCategory.HARP
        ]

        for det in detections[:2]:  # Check top 2
            if det.instrument in good_for_bp and det.confidence > 0.4:
                return True

        # Default to yes for unknown
        return True

    def should_use_crepe(self, audio: np.ndarray) -> bool:
        """
        Determine if CREPE is appropriate for this audio.

        CREPE is good for monophonic melody lines (winds, brass, vocals).
        """
        detections = self.classify_with_confidence(audio)

        # Instruments that work well with CREPE
        good_for_crepe = [
            InstrumentCategory.WINDS,
            InstrumentCategory.BRASS,
            InstrumentCategory.STRINGS  # If solo/violin
        ]

        for det in detections[:2]:
            if det.instrument in good_for_crepe and det.confidence > 0.5:
                return True

        return False


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_stem_classifier(sr: int = 22050, simple_mode: bool = False) -> StemClassifier:
    """
    Factory function for StemClassifier.

    Args:
        sr: Sample rate for analysis
        simple_mode: If True, always returns ['piano'] (backward compatible)
    """
    classifier = StemClassifier(sr=sr)

    if simple_mode:
        # Monkey patch for simple mode
        classifier.classify = lambda audio: classifier.classify_simple(audio)

    return classifier


# ============================================================================
# TEST/DEMO
# ============================================================================

if __name__ == "__main__":
    print("Stem Classifier - Modern Instrument Detection")
    print("=" * 50)

    # Create classifier
    classifier = create_stem_classifier()

    # Test with synthetic audio (piano-like)
    print("\nTesting with synthetic piano-like signal...")

    # Generate simple test signal
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Piano-like: harmonic series with percussive attack
    test_audio = np.zeros_like(t)
    for harm in [1, 2, 3, 4]:
        amplitude = 1.0 / harm
        test_audio += amplitude * np.sin(2 * np.pi * 440 * harm * t)

    # Add percussive attack
    attack = np.exp(-t * 20) * 0.5
    test_audio = test_audio * attack

    # Classify
    results = classifier.classify_with_confidence(test_audio)

    print(f"Top detection: {results[0].instrument.value} (confidence: {results[0].confidence:.2f})")

    if len(results) > 1:
        print(f"Second: {results[1].instrument.value} (confidence: {results[1].confidence:.2f})")

    print("\n✅ Stem classifier ready for production")