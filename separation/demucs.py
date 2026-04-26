#!/usr/bin/env python3
"""
demucs.py — Demucs Stem Separation Wrapper

Reuses your existing Demucs implementation from 4.4.
"""

import os
import sys
import asyncio
import tempfile
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import existing Demucs from 4.4
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("⚠️ Demucs not installed — using fallback")

# Try to import torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed — Demucs disabled")


class DemucsSeparator:
    """
    Demucs stem separator wrapper.
    Reuses the working implementation from Grimlock 4.4.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self._loaded = False
        self.sample_rate = 44100

    def _load_model(self):
        """Load Demucs model (lazy loading)."""
        if self._loaded:
            return self.model

        if not DEMUCS_AVAILABLE or not TORCH_AVAILABLE:
            print("⚠️ Demucs not available — using fallback")
            return None

        try:
            self.device = "cpu"
            print("🔄 Loading Demucs model 'htdemucs'...")
            self.model = get_model("htdemucs")
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"✅ Demucs ready on {self.device}")
        except Exception as e:
            print(f"🔴 Demucs load failed: {e}")
            self.model = None

        return self.model

    def _force_gc_and_clear_cache(self):
        """Force garbage collection and clear GPU cache."""
        import gc
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def release(self) -> None:
        """
        FIX #8 — Explicitly unload Demucs from VRAM/RAM.

        Called from HybridSeparator.release() (or the pipeline directly) after
        separation is complete.  This frees PyTorch tensors so TensorFlow
        (CREPE / SPICE) can claim GPU memory without triggering OOM errors on
        cards with 8–12 GB VRAM.

        Phase discipline (Fix #10):
          Phase 1 Separation — Demucs + Roformer ON
          Phase 2 Pitch       — Demucs + Roformer OFF, TF ON
        """
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        self._force_gc_and_clear_cache()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print("   🧹 Demucs: VRAM cleared (cuda)")
        else:
            print("   🧹 Demucs: model released (CPU mode)")

    async def separate(self, audio_path: Path, sr: int,
                       duration: float = None) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Demucs.

        Args:
            audio_path: Path to the original audio file
            sr:         Sample rate
            duration:   FIX — Maximum seconds to load.  Pass truncate_seconds
                        from the pipeline so Demucs never reads the full file
                        when the user asked for e.g. 60 seconds.

        Returns:
            dict with keys: 'drums', 'bass', 'other', 'vocals'
        """
        # Check if file exists
        if not audio_path.exists():
            print(f"🔴 Audio file not found: {audio_path}")
            y, _ = librosa.load(str(audio_path) if audio_path else "",
                                 sr=sr, duration=duration)
            return {'full_mix': y}

        # Force garbage collection before running
        self._force_gc_and_clear_cache()

        # Load model
        model = self._load_model()

        # Fallback if Demucs not available
        if model is None:
            print("⚠️ Demucs not available — using full mix as 'other'")
            y, _ = librosa.load(str(audio_path), sr=sr, duration=duration)
            return {
                'drums':  np.zeros_like(y),
                'bass':   np.zeros_like(y),
                'vocals': np.zeros_like(y),
                'other':  y
            }

        try:
            # FIX: respect duration so we never load more than truncated length
            wav, _ = librosa.load(str(audio_path), sr=sr, mono=False,
                                   duration=duration)

            if wav is None or len(wav) == 0:
                raise ValueError("Audio buffer is empty")

            # Convert to stereo if mono
            if wav.ndim == 1:
                wav = np.stack([wav, wav])
                print("🎵 Converted mono to stereo")

            print(f"🎵 Audio shape: {wav.shape}, duration: {wav.shape[1] / sr:.1f}s")

            # Convert to tensor
            wav_tensor = torch.from_numpy(wav).float().to(self.device).unsqueeze(0)

            # Run Demucs
            print(f"🔀 Running Demucs separation (30-60 seconds)...")
            with torch.no_grad():
                sources = apply_model(model, wav_tensor, device=self.device,
                                      shifts=1, overlap=0.25)[0]

            print(f"✅ Demucs separation complete")

            # Extract stems
            source_names = ['drums', 'bass', 'other', 'vocals']
            stems = {}

            for i, name in enumerate(source_names):
                if i < len(sources):
                    stem = sources[i].cpu().numpy()
                    if stem.shape[0] > 1:
                        stem = librosa.to_mono(stem)
                    stems[name] = stem.astype(np.float32)
                    print(f"   📀 Stem '{name}': {stem.shape}")

            # Clean up
            self._force_gc_and_clear_cache()

            return stems

        except Exception as e:
            print(f"🔴 Demucs separation error: {e}")
            import traceback
            traceback.print_exc()
            y, _ = librosa.load(str(audio_path), sr=sr, duration=duration)
            return {
                'drums':  np.zeros_like(y),
                'bass':   np.zeros_like(y),
                'vocals': np.zeros_like(y),
                'other':  y
            }


# Singleton instance
_demucs_instance = None


def get_demucs_separator() -> DemucsSeparator:
    """Get singleton Demucs separator instance."""
    global _demucs_instance
    if _demucs_instance is None:
        _demucs_instance = DemucsSeparator()
    return _demucs_instance