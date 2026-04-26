#!/usr/bin/env python3
"""
audio_utils.py — Safe, stable audio handling utilities.

No external dependencies beyond numpy and librosa.
All functions are pure and deterministic.
"""

import numpy as np
import librosa
from typing import Tuple, Optional


# ============================================================================
# BASIC AUDIO SAFETY
# ============================================================================

def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Ensure consistent dtype for all models."""
    if audio is None:
        return np.array([], dtype=np.float32)
    return audio.astype(np.float32, copy=False)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo to mono safely."""
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0).astype(np.float32)


def clamp_audio(audio: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """Prevent model instability from extreme amplitudes."""
    if audio is None or len(audio) == 0:
        return audio
    return np.clip(audio, min_val, max_val)


# ============================================================================
# NORMALIZATION (SAFE VERSION)
# ============================================================================

def peak_normalize(audio: np.ndarray, target: float = 0.99) -> np.ndarray:
    """
    Peak normalization that avoids over-amplifying noise floors.

    Args:
        audio: Input audio array
        target: Target peak amplitude (default 0.99)

    Returns:
        Normalized audio array
    """
    if audio is None or len(audio) == 0:
        return audio

    peak = np.max(np.abs(audio)) + 1e-8
    if peak < 1e-6:
        return audio
    return (audio / peak) * target


def rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    RMS normalization - more stable than peak normalization for ML pipelines.

    Args:
        audio: Input audio array
        target_rms: Target RMS level (default 0.1)

    Returns:
        Normalized audio array
    """
    if audio is None or len(audio) == 0:
        return audio

    rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    return audio * (target_rms / rms)


# ============================================================================
# RESAMPLING
# ============================================================================

def safe_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample with librosa but ensure type stability.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    return librosa.resample(audio.astype(np.float32),
                            orig_sr=orig_sr,
                            target_sr=target_sr)


# ============================================================================
# SILENCE / ENERGY DETECTION
# ============================================================================

def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    """Detect near-silence safely."""
    if audio is None or len(audio) == 0:
        return True
    return np.mean(np.abs(audio)) < threshold


def energy_envelope(audio: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    """Frame-based energy envelope used by multiple detectors."""
    if audio is None or len(audio) == 0:
        return np.array([])

    if len(audio) < frame_size:
        return np.array([np.mean(np.abs(audio))])

    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop)
    return np.mean(np.abs(frames), axis=0)


# ============================================================================
# SEGMENT UTILITIES
# ============================================================================

def extract_window(audio: np.ndarray, sr: int, time: float,
                   window_ms: float = 50.0) -> np.ndarray:
    """
    Extract a centered window around a time index.

    Args:
        audio: Full audio array
        sr: Sample rate
        time: Center time in seconds
        window_ms: Window duration in milliseconds

    Returns:
        Windowed audio segment
    """
    if audio is None or len(audio) == 0:
        return np.array([])

    half = int((window_ms / 1000.0) * sr / 2)
    center = int(time * sr)

    start = max(0, center - half)
    end = min(len(audio), center + half)

    return audio[start:end]


def safe_onset_slice(audio: np.ndarray, sr: int, onset_time: float,
                     post_ms: float = 200.0) -> np.ndarray:
    """
    Slice audio after onset for decay analysis.

    Args:
        audio: Full audio array
        sr: Sample rate
        onset_time: Onset time in seconds
        post_ms: Duration after onset in milliseconds

    Returns:
        Audio segment from onset to onset+post_ms
    """
    if audio is None or len(audio) == 0:
        return np.array([])

    start = int(onset_time * sr)
    end = min(len(audio), start + int(sr * post_ms / 1000.0))
    return audio[start:end]


# ============================================================================
# DEBUG / SAFETY UTILITIES
# ============================================================================

def validate_audio(audio: np.ndarray, name: str = "audio") -> np.ndarray:
    """
    Central guard against silent pipeline corruption.

    Args:
        audio: Input audio array
        name: Name for error messages

    Returns:
        Validated audio array
    """
    if audio is None:
        raise ValueError(f"{name} is None")

    audio = ensure_float32(audio)

    if np.any(np.isnan(audio)):
        audio = np.nan_to_num(audio)
        print(f"⚠️ {name}: NaNs detected and replaced")

    if np.any(np.isinf(audio)):
        audio = np.nan_to_num(audio)
        print(f"⚠️ {name}: Infs detected and replaced")

    return audio