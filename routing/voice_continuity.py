#!/usr/bin/env python3
"""
voice_continuity.py — Keep melodic lines together

Prevents a saxophone run from being split across multiple voices.
"""

from typing import List
from order_types import Note


class VoiceContinuity:
    """
    Enforces that notes close in time and pitch belong to the same voice.
    """

    def enforce(self, notes: List[Note]) -> List[Note]:
        """
        Assign voice IDs based on continuity.
        """
        if len(notes) < 2:
            return notes

        # Sort by time
        notes.sort(key=lambda x: x.start)

        current_voice = 0
        notes[0].voice_id = current_voice

        for i in range(1, len(notes)):
            prev = notes[i - 1]
            curr = notes[i]

            time_gap = curr.start - prev.end
            pitch_change = abs(curr.pitch - prev.pitch)

            # If close in time and pitch, same voice
            if time_gap < 0.2 and pitch_change < 5:
                curr.voice_id = prev.voice_id
            else:
                # New phrase or different voice
                current_voice += 1
                curr.voice_id = current_voice

        return notes