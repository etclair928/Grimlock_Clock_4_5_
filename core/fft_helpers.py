#!/usr/bin/env python3
"""
fft_helpers.py — Spectral analysis utilities.

Stable FFT and spectrogram functions for feature extraction.
"""

import numpy as np
import librosa
from typing import Tuple, Optional


# ============================================================================
# CORE FFT TRANSFORMS
# ============================================================================

def compute_fft(audio: np.ndarray) -> np.ndarray:
    """Raw FFT magnitude spectrum."""
    if audio is None or len(audio) == 0:
        return np.array([])
    return np.abs(np.fft.rfft(audio))


def compute_stft(audio: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Short-time Fourier transform (magnitude)."""
    if audio is None or len(audio) == 0:
        return np.array([])
    return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))


def compute_mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 64,
                            fmin: float = 20, fmax: Optional[float] = None) -> np.ndarray:
    """
    Mel spectrogram (core for NMF + classification).

    Args:
        audio: Input audio array
        sr: Sample rate
        n_mels: Number of mel bands
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz, defaults to sr/2)

    Returns:
        Mel spectrogram in dB scale
    """
    if audio is None or len(audio) == 0:
        return np.array([])

    if fmax is None:
        fmax = sr // 2

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    return librosa.power_to_db(mel, ref=np.max)


# ============================================================================
# FEATURE EXTRACTION HELPERS
# ============================================================================

def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Global spectral centroid."""
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))


def spectral_band_energy(audio: np.ndarray, sr: int, low: float, high: float) -> float:
    """
    Energy in a frequency band.

    Args:
        audio: Input audio array
        sr: Sample rate
        low: Low frequency bound (Hz)
        high: High frequency bound (Hz)

    Returns:
        Total energy in the specified band
    """
    if audio is None or len(audio) == 0:
        return 0.0

    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0

    return float(np.sum(stft[mask, :]))


def zero_crossing_rate(audio: np.ndarray, frame_length: int = 2048, hop: int = 512) -> float:
    """Zero-crossing rate (average across frames)."""
    if audio is None or len(audio) == 0:
        return 0.0
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length,
                                             hop_length=hop)
    return float(np.mean(zcr))


# ============================================================================
# NORMALIZATION (SPECTRAL SAFE ZONES)
# ============================================================================

def normalize_spectrum(spec: np.ndarray) -> np.ndarray:
    """Avoid division instability in ML pipelines."""
    if spec is None or spec.size == 0:
        return spec
    return spec / (np.max(spec) + 1e-8)


def log_compress(spec: np.ndarray) -> np.ndarray:
    """Dynamic range compression."""
    if spec is None or spec.size == 0:
        return spec
    return np.log1p(spec)


# ============================================================================
# PEAK ANALYSIS
# ============================================================================

def detect_spectral_peaks(spectrum: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Simple peak mask.

    Args:
        spectrum: FFT magnitude spectrum
        threshold: Relative threshold (0-1) of max peak

    Returns:
        Indices of detected peaks
    """
    if spectrum is None or spectrum.size == 0:
        return np.array([])
    return np.where(spectrum > threshold * np.max(spectrum))[0]


# ============================================================================
# HARMONIC ANALYSIS
# ============================================================================

def harmonic_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Rough harmonic vs noise ratio.

    Useful for:
    - drum rejection (low ratio = noise/drum)
    - pitch confidence weighting (high ratio = pitched)

    Returns:
        float: 0.0 (pure noise) to 1.0 (pure harmonic)
    """
    if audio is None or len(audio) == 0:
        return 0.5

    centroid = spectral_centroid(audio, sr)
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    # Normalize: lower flatness = more harmonic
    # Flatness of 0.1 = harmonic, 1.0 = noise
    harmonic_score = 1.0 - min(1.0, flatness)

    # Also consider centroid: low centroid could be noise or bass
    # Bass is harmonic but has low centroid
    if centroid < 150:
        harmonic_score = max(harmonic_score, 0.5)

    return float(harmonic_score)