#!/usr/bin/env python3
"""
deep_analysis.py — Second-Pass Perceptual Magnifier

When confidence is low, slow down the audio to increase temporal resolution.
Results are time-corrected and merged with original results.

"Listen slower. See more. Then correct."

Process:
1. Time-stretch audio to 0.66x speed
2. Re-run full analysis pipeline
3. Remap times back to original
4. Merge with original results (weighted: 0.6 original / 0.4 slow)
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

# Phase 1 imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from order_types import DrumHit, Note, RhythmInfo, TranscriptionResult, DrumType, SourceType


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DeepAnalysisConfig:
    """Configuration for deep analysis."""

    # Time stretch factor (0.66x = 50% slower)
    stretch_factor: float = 0.66

    # Merge weights (original : slow)
    original_weight: float = 0.6
    slow_weight: float = 0.4

    # Merge tolerance (ms) — hits within this distance are considered same event
    merge_tolerance_ms: float = 30.0

    # Confidence boost for slow-pass hits (they are less trusted)
    slow_confidence_penalty: float = 0.85

    # Maximum number of deep analysis passes (avoid infinite loops)
    max_passes: int = 2

    # Only trigger if confidence below threshold (handled by router)
    trigger_threshold: float = 0.65


# ============================================================================
# DEEP ANALYSIS ENGINE (FIXED)
# ============================================================================

class DeepAnalysisEngine:
    """
    Second-pass analysis at slower speed.

    Time-stretching increases temporal resolution, making it easier to:
    - Separate fast drum patterns (ride cymbals)
    - Detect ghost notes
    - Resolve flams and fast runs
    """

    def __init__(self, config: Optional[DeepAnalysisConfig] = None):
        self.config = config or DeepAnalysisConfig()
        self._pass_count = 0

    def time_stretch(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Stretch audio in time without changing pitch.

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            (stretched_audio, original_sr) — sr unchanged
        """
        stretched = librosa.effects.time_stretch(audio, rate=self.config.stretch_factor)
        return stretched, sr

    def remap_time(self, slow_time: float) -> float:
        """
        Remap time from slow-pass to original time axis.

        Args:
            slow_time: Time in stretched audio

        Returns:
            Time in original audio
        """
        return slow_time * self.config.stretch_factor

    def remap_hits(self, hits: List[DrumHit]) -> List[DrumHit]:
        """
        Remap drum hit times from slow-pass to original.

        Args:
            hits: Drum hits from slow-pass

        Returns:
            Drum hits with remapped times and confidence penalty
        """
        remapped = []
        for hit in hits:
            new_hit = DrumHit(
                time=self.remap_time(hit.time),
                drum_type=hit.drum_type,
                confidence=hit.confidence * self.config.slow_confidence_penalty,
                velocity=hit.velocity,
                source=hit.source,
                beat_position=hit.beat_position,
                grid_deviation_ms=hit.grid_deviation_ms,
                is_inferred=hit.is_inferred
            )
            remapped.append(new_hit)
        return remapped

    def remap_notes(self, notes: List[Note]) -> List[Note]:
        """
        Remap note times from slow-pass to original.

        Args:
            notes: Notes from slow-pass

        Returns:
            Notes with remapped times and confidence penalty
        """
        remapped = []
        for note in notes:
            new_note = Note(
                pitch=note.pitch,
                start=self.remap_time(note.start),
                end=self.remap_time(note.end),
                velocity=note.velocity,
                confidence=note.confidence * self.config.slow_confidence_penalty,
                instrument=note.instrument,
                source=note.source,
                voice_id=note.voice_id
            )
            remapped.append(new_note)
        return remapped

    def merge_hits(self, original_hits: List[DrumHit],
                   slow_hits: List[DrumHit]) -> List[DrumHit]:
        """
        Merge original and slow-pass drum hits.

        FIXED: No more list.remove() errors. Uses index-based replacement.
        """
        tolerance = self.config.merge_tolerance_ms / 1000.0

        # Start with a copy of original hits (as list of dicts for easier manipulation)
        merged = []
        for hit in original_hits:
            merged.append({
                'time': hit.time,
                'drum_type': hit.drum_type,
                'confidence': hit.confidence,
                'velocity': hit.velocity,
                'source': hit.source,
                'beat_position': hit.beat_position,
                'grid_deviation_ms': hit.grid_deviation_ms,
                'is_inferred': hit.is_inferred,
                'is_original': True,
                'original_hit': hit
            })

        # Process slow hits
        for slow_hit in slow_hits:
            is_duplicate = False

            # Check if this slow hit matches any existing hit
            for i, m in enumerate(merged):
                if abs(m['time'] - slow_hit.time) <= tolerance:
                    is_duplicate = True
                    # If slow hit has higher confidence, replace
                    if slow_hit.confidence > m['confidence']:
                        merged[i] = {
                            'time': slow_hit.time,
                            'drum_type': slow_hit.drum_type,
                            'confidence': slow_hit.confidence,
                            'velocity': slow_hit.velocity,
                            'source': slow_hit.source,
                            'beat_position': slow_hit.beat_position,
                            'grid_deviation_ms': slow_hit.grid_deviation_ms,
                            'is_inferred': slow_hit.is_inferred,
                            'is_original': False,
                            'original_hit': slow_hit
                        }
                    break

            # If not a duplicate, add it
            if not is_duplicate:
                merged.append({
                    'time': slow_hit.time,
                    'drum_type': slow_hit.drum_type,
                    'confidence': slow_hit.confidence,
                    'velocity': slow_hit.velocity,
                    'source': slow_hit.source,
                    'beat_position': slow_hit.beat_position,
                    'grid_deviation_ms': slow_hit.grid_deviation_ms,
                    'is_inferred': slow_hit.is_inferred,
                    'is_original': False,
                    'original_hit': slow_hit
                })

        # Convert back to DrumHit objects
        result = []
        for m in merged:
            if m['is_original']:
                result.append(m['original_hit'])
            else:
                # Create new DrumHit
                result.append(DrumHit(
                    time=m['time'],
                    drum_type=m['drum_type'],
                    confidence=m['confidence'],
                    velocity=m['velocity'],
                    source=m['source'],
                    beat_position=m['beat_position'],
                    grid_deviation_ms=m['grid_deviation_ms'],
                    is_inferred=m['is_inferred']
                ))

        # Sort by time
        result.sort(key=lambda x: x.time)

        # Final deduplication within tolerance
        final = []
        for hit in result:
            if not final or abs(hit.time - final[-1].time) > tolerance:
                final.append(hit)
            elif hit.confidence > final[-1].confidence:
                final[-1] = hit

        return final

    def merge_notes(self, original_notes: List[Note],
                    slow_notes: List[Note]) -> List[Note]:
        """
        Merge original and slow-pass pitched notes.

        Similar to merge_hits but with pitch consideration.
        """
        tolerance = self.config.merge_tolerance_ms / 1000.0

        # Start with a copy of original notes (as dicts)
        merged = []
        for note in original_notes:
            merged.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity,
                'confidence': note.confidence,
                'instrument': note.instrument,
                'source': note.source,
                'voice_id': note.voice_id,
                'is_original': True,
                'original_note': note
            })

        # Process slow notes
        for slow_note in slow_notes:
            is_duplicate = False

            for i, m in enumerate(merged):
                if (abs(m['start'] - slow_note.start) <= tolerance and
                        m['pitch'] == slow_note.pitch):
                    is_duplicate = True
                    if slow_note.confidence > m['confidence']:
                        merged[i] = {
                            'pitch': slow_note.pitch,
                            'start': slow_note.start,
                            'end': slow_note.end,
                            'velocity': slow_note.velocity,
                            'confidence': slow_note.confidence,
                            'instrument': slow_note.instrument,
                            'source': slow_note.source,
                            'voice_id': slow_note.voice_id,
                            'is_original': False,
                            'original_note': slow_note
                        }
                    break

            if not is_duplicate:
                merged.append({
                    'pitch': slow_note.pitch,
                    'start': slow_note.start,
                    'end': slow_note.end,
                    'velocity': slow_note.velocity,
                    'confidence': slow_note.confidence,
                    'instrument': slow_note.instrument,
                    'source': slow_note.source,
                    'voice_id': slow_note.voice_id,
                    'is_original': False,
                    'original_note': slow_note
                })

        # Convert back to Note objects
        result = []
        for m in merged:
            if m['is_original']:
                result.append(m['original_note'])
            else:
                result.append(Note(
                    pitch=m['pitch'],
                    start=m['start'],
                    end=m['end'],
                    velocity=m['velocity'],
                    confidence=m['confidence'],
                    instrument=m['instrument'],
                    source=m['source'],
                    voice_id=m['voice_id']
                ))

        result.sort(key=lambda x: x.start)
        return result

    def merge_rhythm(self, original_rhythm: RhythmInfo,
                     slow_rhythm: RhythmInfo) -> RhythmInfo:
        """
        Merge original and slow-pass rhythm info.

        Weighted average of tempo and confidence.
        """
        w_orig = self.config.original_weight
        w_slow = self.config.slow_weight

        merged_tempo = (original_rhythm.tempo * w_orig + slow_rhythm.tempo * w_slow)
        merged_confidence = (original_rhythm.confidence * w_orig +
                             slow_rhythm.confidence * w_slow)

        # Use original beat times (more reliable)
        return RhythmInfo(
            tempo=merged_tempo,
            beat_times=original_rhythm.beat_times,
            downbeats=original_rhythm.downbeats,
            time_signature=original_rhythm.time_signature,
            beats_per_bar=original_rhythm.beats_per_bar,
            confidence=merged_confidence,
            grid=original_rhythm.grid
        )

    async def run_deep_analysis(self, audio: np.ndarray, sr: int,
                                analysis_func) -> Tuple[Any, Any, Any]:
        """
        Run deep analysis with time-stretching.

        Args:
            audio: Original audio array
            sr: Sample rate
            analysis_func: Async function that takes (audio, sr) and returns
                          (drum_hits, notes, rhythm_info)

        Returns:
            Tuple of (drum_hits, notes, rhythm_info) from merged analysis
        """
        self._pass_count += 1

        if self._pass_count > self.config.max_passes:
            print(f"⚠️ Max deep analysis passes ({self.config.max_passes}) reached")
            return await analysis_func(audio, sr)

        # Step 1: Stretch audio
        print(f"🐢 Deep analysis pass {self._pass_count}: stretching to {self.config.stretch_factor}x speed")
        stretched_audio, _ = self.time_stretch(audio, sr)

        # Step 2: Run analysis on stretched audio
        slow_drums, slow_notes, slow_rhythm = await analysis_func(stretched_audio, sr)

        # Step 3: Remap times back
        slow_drums = self.remap_hits(slow_drums)
        slow_notes = self.remap_notes(slow_notes)

        # Step 4: Run original analysis again (for fresh comparison)
        orig_drums, orig_notes, orig_rhythm = await analysis_func(audio, sr)

        # Step 5: Merge with original results
        merged_drums = self.merge_hits(orig_drums, slow_drums)
        merged_notes = self.merge_notes(orig_notes, slow_notes)
        merged_rhythm = self.merge_rhythm(orig_rhythm, slow_rhythm)

        print(f"🐢 Deep analysis complete: "
              f"drums {len(orig_drums)}→{len(merged_drums)}, "
              f"notes {len(orig_notes)}→{len(merged_notes)}")

        return merged_drums, merged_notes, merged_rhythm

    def reset(self):
        """Reset pass counter for new file."""
        self._pass_count = 0


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Test Deep Analysis Engine")
    parser.add_argument("audio_file", help="Path to audio file")
    args = parser.parse_args()


    async def mock_analysis(audio, sr):
        """Mock analysis function for testing."""
        # Simulate some hits
        duration = len(audio) / sr
        hits = []
        notes = []

        for i in range(int(duration * 2)):
            t = i * 0.5
            hits.append(DrumHit(
                time=t,
                drum_type=DrumType.KICK,
                confidence=0.7,
                velocity=80
            ))

        rhythm = RhythmInfo(
            tempo=120.0,
            beat_times=list(np.arange(0, duration, 0.5)),
            downbeats=list(np.arange(0, duration, 2.0)),
            time_signature="4/4",
            confidence=0.65
        )

        return hits, notes, rhythm


    async def test():
        # Load audio
        audio, sr = librosa.load(args.audio_file, sr=22050)

        # Create deep analysis engine
        engine = DeepAnalysisEngine()

        # Run deep analysis
        drums, notes, rhythm = await engine.run_deep_analysis(audio, sr, mock_analysis)

        print(f"\n{'=' * 60}")
        print(f"Deep Analysis Test")
        print(f"{'=' * 60}")
        print(f"Original duration: {len(audio) / sr:.2f}s")
        print(f"Pass count: {engine._pass_count}")
        print(f"Merged drum hits: {len(drums)}")
        print(f"Merged tempo: {rhythm.tempo:.1f} BPM")

        if drums:
            print(f"\nFirst 10 drum hits:")
            for hit in drums[:10]:
                print(f"  {hit.time:.3f}s - {hit.drum_type.value} (conf={hit.confidence:.2f})")


    asyncio.run(test())