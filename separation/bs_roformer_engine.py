#!/usr/bin/env python3
import os
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Dict

# Rename the import to be explicit and avoid shadowing
try:
    from bs_roformer import BSRoformer as BSRoformerModel

    BS_LIB_FOUND = True
except ImportError:
    BS_LIB_FOUND = False


class BSRoformerSeparator:
    def __init__(self):
        self.model = None
        self._loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if self._loaded: return self.model
        if not BS_LIB_FOUND: return None

        try:
            # Match architecture for bs_roformer_ep_317_sdr_12.9755.ckpt
            self.model = BSRoformerModel(
                dim=512,
                depth=12,
                stereo=True,
                num_stems=6,  # Standard for this checkpoint
                time_transformer_depth=1,
                freq_transformer_depth=1
            )

            weights = Path("./models/bs_roformer_ep_317_sdr_12.9755.ckpt")
            if weights.exists():
                print(f"   [ROFORMER] Loading weights from {weights}")
                state_dict = torch.load(weights, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self._loaded = True
        except Exception as e:
            print(f"   ❌ Roformer load failed: {e}")
            self.model = None
        return self.model

    async def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        model = self._load_model()
        if model is None: return {"other": audio}

        # Convert to [1, 2, Samples] for the Transformer
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        mix = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Standard forward pass
            stems = model(mix)

        return {name: tensor.squeeze(0).cpu().numpy() for name, tensor in stems.items()}

    def release(self) -> None:
        """
        FIX #8 — Explicitly unload BS-Roformer from VRAM after separation.

        Call this from HybridSeparator.release() (or directly from the pipeline)
        before the Pitch Intelligence phase begins.  Prevents OOM errors when
        TensorFlow (CREPE / SPICE) tries to claim GPU memory while PyTorch
        models are still resident.
        """
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   🧹 BS-Roformer: VRAM cleared")
        else:
            print("   🧹 BS-Roformer: model released (CPU mode)")