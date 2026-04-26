#!/usr/bin/env python3
"""
time_utils.py — Timing utilities for rhythm analysis.

Grid building, snapping, tempo calculations, and humanization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ============================================================================
# BASIC TIME CONVERSIONS
# ============================================================================

def bpm_to_seconds_per_beat(bpm: float) -> float:
    """Convert BPM to seconds per beat."""
    return 60.0 / max(bpm, 1e-6)


def seconds_per_beat_to_bpm(spb: float) -> float:
    """Convert seconds per beat to BPM."""
    return 60.0 / max(spb, 1e-6)


def beats_to_seconds(beats: float, bpm: float) -> float:
    """Convert beats to seconds."""
    return beats * bpm_to_seconds_per_beat(bpm)


def seconds_to_beats(seconds: float, bpm: float) -> float:
    """Convert seconds to beats."""
    return seconds / bpm_to_seconds_per_beat(bpm)


# ============================================================================
# GRID GENERATION
# ============================================================================

def build_beat_grid(beat_times: List[float], bpm: float,
                    beats_per_bar: int = 4) -> Dict[str, np.ndarray]:
    """
    Build structured grid: beat_times, bar indices, beat positions within bar.

    Args:
        beat_times: List of detected beat times
        bpm: Tempo in BPM
        beats_per_bar: Number of beats per bar (4 = 4/4, 3 = 3/4, etc.)

    Returns:
        Dict with 'grid', 'bar_map', 'beat_map' keys
    """
    if not beat_times:
        return {
            "grid": np.array([]),
            "bar_map": np.array([]),
            "beat_map": np.array([])
        }

    beat_times = np.array(sorted(beat_times))
    grid = beat_times.copy()
    beat_map = np.arange(len(grid)) % beats_per_bar
    bar_map = np.arange(len(grid)) // beats_per_bar

    return {
        "grid": grid,
        "bar_map": bar_map,
        "beat_map": beat_map
    }


def build_subdivision_grid(beat_times: List[float], tempo: float,
                           subdivisions_per_beat: int = 4) -> np.ndarray:
    """
    Build fine subdivision grid (16th notes, triplets, etc.).

    Args:
        beat_times: List of beat times
        tempo: Tempo in BPM
        subdivisions_per_beat: Number of subdivisions per beat (4 = 16th notes)

    Returns:
        Sorted array of subdivision times
    """
    if not beat_times:
        return np.array([])

    beat_duration = 60.0 / tempo
    sub_duration = beat_duration / subdivisions_per_beat

    grid = []
    for beat in beat_times:
        for sub in range(subdivisions_per_beat):
            grid.append(beat + sub * sub_duration)

    return np.array(sorted(set(grid)))


# ============================================================================
# GRID SNAPPING (CORE OF RHYTHM ENGINE)
# ============================================================================

def snap_to_grid(time: float, grid: np.ndarray, tolerance_ms: float = 65.0) -> Tuple[bool, float, int]:
    """
    Snap a time value to nearest beat grid.

    Args:
        time: Time in seconds
        grid: Array of grid times
        tolerance_ms: Tolerance in milliseconds

    Returns:
        Tuple: (is_valid, snapped_time, index)
    """
    if grid is None or len(grid) == 0:
        return True, time, -1

    tolerance = tolerance_ms / 1000.0

    idx = int(np.argmin(np.abs(grid - time)))
    dist = abs(grid[idx] - time)

    if dist <= tolerance:
        return True, float(grid[idx]), idx

    return False, float(time), -1


def snap_to_grid_with_penalty(time: float, grid: np.ndarray,
                              tolerance_ms: float = 65.0,
                              penalty: float = 0.5) -> Tuple[float, int, float]:
    """
    Snap to grid with confidence penalty for off-grid hits.

    Returns:
        Tuple: (snapped_time, index, confidence_multiplier)
    """
    if grid is None or len(grid) == 0:
        return time, -1, 1.0

    tolerance = tolerance_ms / 1000.0

    idx = int(np.argmin(np.abs(grid - time)))
    dist = abs(grid[idx] - time)

    if dist <= tolerance:
        return float(grid[idx]), idx, 1.0
    else:
        return float(time), -1, penalty


# ============================================================================
# SWING / HUMANIZATION
# ============================================================================

def apply_swing(beat_times: List[float], swing_amount: float = 0.15) -> List[float]:
    """
    Apply classic jazz swing (delays off-beats).

    Args:
        beat_times: List of beat times
        swing_amount: 0.0 (none) to 0.33 (heavy swing)

    Returns:
        Swung beat times
    """
    if not beat_times or swing_amount <= 0:
        return beat_times.copy()

    beat_times = sorted(beat_times)
    swung = []

    for i, t in enumerate(beat_times):
        if i % 2 == 1 and i > 0:
            # off-beat delay
            delay = (beat_times[i] - beat_times[i - 1]) * swing_amount
            swung.append(t + delay)
        else:
            swung.append(t)

    return swung


# ============================================================================
# BAR DETECTION
# ============================================================================

def group_into_bars(beat_times: List[float], beats_per_bar: int = 4) -> List[List[float]]:
    """Group beats into bars."""
    if not beat_times:
        return []

    bars = []
    for i in range(0, len(beat_times), beats_per_bar):
        bars.append(beat_times[i:i + beats_per_bar])

    return bars


# ============================================================================
# TIME ALIGNMENT HELPERS
# ============================================================================

def find_nearest_event(time: float, events: List[float],
                       tolerance: float = 0.05) -> Optional[float]:
    """Find nearest event within tolerance."""
    if not events:
        return None

    nearest = min(events, key=lambda x: abs(x - time))

    if abs(nearest - time) <= tolerance:
        return nearest

    return None


def compute_ioi(events: List[float]) -> np.ndarray:
    """Inter-onset intervals (tempo backbone feature)."""
    if len(events) < 2:
        return np.array([])
    return np.diff(np.array(sorted(events)))


# ============================================================================
# FEATURE ALIGNMENT (USED BY DRUM + PITCH + NMF)
# ============================================================================

def align_to_grid_events(events: List[Dict], grid: np.ndarray,
                         tolerance_ms: float = 65.0) -> List[Dict]:
    """
    Align detected events to rhythmic grid.

    Args:
        events: List of dicts with 'time' key
        grid: Array of grid times
        tolerance_ms: Tolerance in milliseconds

    Returns:
        Events with added 'on_grid', 'grid_index', 'snapped_time' fields
    """
    tolerance = tolerance_ms / 1000.0
    aligned = []

    for e in events:
        t = e.get("time", 0)
        if len(grid) == 0:
            e["on_grid"] = False
            e["grid_index"] = -1
            e["snapped_time"] = t
            aligned.append(e)
            continue

        idx = int(np.argmin(np.abs(grid - t)))
        dist = abs(grid[idx] - t)

        if dist <= tolerance:
            e["on_grid"] = True
            e["grid_index"] = idx
            e["snapped_time"] = float(grid[idx])
        else:
            e["on_grid"] = False
            e["grid_index"] = -1
            e["snapped_time"] = t

        aligned.append(e)

    return aligned


# ============================================================================
# TEMPO STABILITY CHECK
# ============================================================================

def estimate_tempo_stability(ioi: np.ndarray) -> float:
    """
    Measure rhythmic consistency (0-1).
    Higher = more stable tempo.
    """
    if len(ioi) < 2:
        return 0.0

    cv = np.std(ioi) / (np.mean(ioi) + 1e-6)  # coefficient of variation
    return float(1.0 / (1.0 + cv))  # Convert to 0-1 where 1 = perfectly stable


# ============================================================================
# HUMANIZATION (VERY IMPORTANT FOR OUTPUT QUALITY)
# ============================================================================

def add_micro_timing_variation(events: List[Dict], amount_ms: float = 8.0) -> List[Dict]:
    """
    Add subtle human timing drift.

    Args:
        events: List of dicts with 'time' key
        amount_ms: Maximum jitter in milliseconds

    Returns:
        Events with jittered times
    """
    if not events:
        return []

    jitter = amount_ms / 1000.0

    for e in events:
        e["original_time"] = e["time"]
        e["time"] += np.random.uniform(-jitter, jitter)

    return events