#!/usr/bin/env python3
"""
core/ — Foundation utilities for Grimlock 4.5

Stable, reusable functions for audio handling, FFT, and time operations.
"""

from .audio_utils import (
    ensure_float32,
    to_mono,
    clamp_audio,
    peak_normalize,
    rms_normalize,
    safe_resample,
    is_silent,
    energy_envelope,
    extract_window,
    safe_onset_slice,
    validate_audio,
)

from .fft_helpers import (
    compute_fft,
    compute_stft,
    compute_mel_spectrogram,
    spectral_centroid,
    spectral_band_energy,
    zero_crossing_rate,
    normalize_spectrum,
    log_compress,
    detect_spectral_peaks,
    harmonic_ratio,
)

from .time_utils import (
    bpm_to_seconds_per_beat,
    seconds_per_beat_to_bpm,
    beats_to_seconds,
    seconds_to_beats,
    build_beat_grid,
    snap_to_grid,
    apply_swing,
    group_into_bars,
    find_nearest_event,
    compute_ioi,
    align_to_grid_events,
    estimate_tempo_stability,
    add_micro_timing_variation,
)

__all__ = [
    # audio_utils
    "ensure_float32", "to_mono", "clamp_audio", "peak_normalize",
    "rms_normalize", "safe_resample", "is_silent", "energy_envelope",
    "extract_window", "safe_onset_slice", "validate_audio",
    # fft_helpers
    "compute_fft", "compute_stft", "compute_mel_spectrogram",
    "spectral_centroid", "spectral_band_energy", "zero_crossing_rate",
    "normalize_spectrum", "log_compress", "detect_spectral_peaks",
    "harmonic_ratio",
    # time_utils
    "bpm_to_seconds_per_beat", "seconds_per_beat_to_bpm",
    "beats_to_seconds", "seconds_to_beats", "build_beat_grid",
    "snap_to_grid", "apply_swing", "group_into_bars", "find_nearest_event",
    "compute_ioi", "align_to_grid_events", "estimate_tempo_stability",
    "add_micro_timing_variation",
]