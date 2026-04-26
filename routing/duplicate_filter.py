#!/usr/bin/env python3
"""
duplicate_filter.py — Remove duplicate notes from bleed

Prevents vocals from being transcribed twice (once in vocals stem, once in winds).
"""

from typing import List
from order_types import Note


class DuplicateFilter:
    """
    Removes notes from secondary stems that already exist in primary stems.
    """

    def filter(self, primary_notes: List[Note], secondary_notes: List[Note],
               tolerance_ms: float = 50.0) -> List[Note]:
        """
        Remove notes from secondary that are duplicates of primary.
        """
        if not primary_notes or not secondary_notes:
            return secondary_notes

        tolerance = tolerance_ms / 1000.0

        filtered = []
        for sec_note in secondary_notes:
            is_duplicate = False
            for pri_note in primary_notes:
                time_diff = abs(sec_note.start - pri_note.start)
                pitch_diff = abs(sec_note.pitch - pri_note.pitch)

                if time_diff < tolerance and pitch_diff <= 1:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(sec_note)

        return filtered