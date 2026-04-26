#!/usr/bin/env python3
"""
hybrid_separator.py — Real Demucs + BS-Roformer Hybrid Separation

FIX APPLIED:
- separate() now accepts `duration` parameter and passes it everywhere
- Demucs receives the duration limit so it never processes more than truncated audio
- Fallback librosa.load() calls use `duration` instead of hardcoded 60
- BS-Roformer receives the already-truncated other_stem (no secondary truncation needed)
- Aggressive memory cleanup between Demucs and BS-Roformer (Fix 2)
- Added truncate_for_roformer config (extra safety cap on BS-Roformer input)
"""

import os
import sys
import gc
import time
import asyncio
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field

# Fix imports with fallbacks
try:
    from separation.demucs import get_demucs_separator
except ImportError:
    print("⚠️ Demucs separator not found - using mock")
    get_demucs_separator = None

try:
    from separation.bs_roformer_engine import BSRoformerSeparator
except ImportError:
    print("⚠️ BS-Roformer not found - using mock")
    BSRoformerSeparator = None

try:
    from core.fft_helpers import spectral_centroid
except ImportError:
    def spectral_centroid(audio, sr, hop_length=512):
        try:
            return librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        except Exception:
            return np.array([500.0])


@dataclass
class SeparationConfig:
    """Configuration for hybrid separator"""
    demucs_timeout: float = 120.0
    roformer_timeout: float = 60.0
    enable_roformer: bool = True
    # FIX: cap the 'other' stem sent to BS-Roformer (seconds).
    # Even if Demucs produces a longer stem, BS-Roformer only sees this many
    # seconds.  Set to 0 to disable.
    truncate_for_roformer: int = 30
    use_streaming: bool = False
    chunk_seconds: float = 30.0
    enable_artefact_detection: bool = True
    release_memory_early: bool = True


class HybridSeparator:
    """
    Hybrid separator: Demucs first, then BS-Roformer on the 'other' stem.

    Key fix: separate() accepts a `duration` parameter so it never
    re-loads more audio than the pipeline has already truncated to.
    """

    REQUIRED_STEMS = ('drums', 'bass', 'vocals', 'piano', 'guitar',
                      'strings', 'winds', 'other_residual')

    def __init__(self, config: Optional[SeparationConfig] = None,
                 state: Any = None,
                 progress_callback: Optional[Callable[[int, str], None]] = None):
        self.config = config or SeparationConfig()
        self.state = state
        self.progress_callback = progress_callback
        self._demucs = None
        self._roformer = None
        self._demucs_loaded = False
        self._roformer_loaded = False

    def _update_progress(self, percent: int, message: str):
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception as e:
                print(f"⚠️ Progress callback failed: {e}")

    def _get_demucs(self):
        if not self._demucs_loaded:
            if get_demucs_separator is None:
                raise ImportError("Demucs separator not available")
            self._update_progress(12, "Loading Demucs model...")
            self._demucs = get_demucs_separator()
            self._demucs_loaded = True
        return self._demucs

    def _get_roformer(self):
        if not self._roformer_loaded:
            if BSRoformerSeparator is None:
                print("⚠️ BS-Roformer not available")
                return None
            self._update_progress(15, "Loading BS-Roformer model...")
            self._roformer = BSRoformerSeparator()
            self._roformer_loaded = True
        return self._roformer

    def _validate_audio(self, audio_path: Path) -> Tuple[bool, str]:
        if not audio_path.exists():
            return False, f"File not found: {audio_path}"
        if audio_path.stat().st_size == 0:
            return False, "File is empty"
        try:
            y, sr = librosa.load(str(audio_path), duration=1.0, sr=None)
            if len(y) == 0:
                return False, "No audio data"
            if np.max(np.abs(y)) < 0.001:
                return False, "Audio is practically silent"
        except Exception as e:
            return False, f"Cannot load audio: {e}"
        return True, "OK"

    def _create_fallback_stems(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Create fallback stems when Demucs fails completely."""
        return {
            'drums': y, 'bass': y, 'vocals': y, 'piano': y, 'other': y,
            'guitar': np.zeros_like(y), 'strings': np.zeros_like(y),
            'winds': np.zeros_like(y), 'other_residual': np.zeros_like(y)
        }

    def _ensure_demucs_stems(self, demucs_stems: Dict,
                              audio_path: Path, sr: int,
                              duration: Optional[float]) -> Dict:
        """Ensure all required Demucs stems exist with consistent length."""
        required = ['drums', 'bass', 'vocals', 'other']
        missing = [s for s in required
                   if s not in demucs_stems or demucs_stems[s] is None]
        for stem in missing:
            demucs_stems[stem] = np.zeros(1)
        if missing:
            print(f"   ⚠️ Demucs missing stems: {missing}")

        max_len = max(
            (len(demucs_stems[s]) for s in required
             if demucs_stems.get(s) is not None and len(demucs_stems[s]) > 0),
            default=0
        )

        if max_len == 0:
            # FIX: use duration here — NOT hardcoded 60
            load_dur = duration if duration and duration > 0 else None
            print(f"   ⚠️ No valid stem lengths — loading fallback audio "
                  f"(duration={load_dur}s)")
            y, _ = librosa.load(str(audio_path), sr=sr, duration=load_dur)
            max_len = len(y)
            for stem in required:
                demucs_stems[stem] = y

        # Trim/pad all stems to same length
        for stem in required:
            arr = demucs_stems.get(stem)
            if arr is not None and len(arr) != max_len:
                if len(arr) < max_len:
                    demucs_stems[stem] = np.pad(arr, (0, max_len - len(arr)))
                else:
                    demucs_stems[stem] = arr[:max_len]

        return demucs_stems

    def _ensure_all_stems(self, results: Dict[str, np.ndarray],
                           reference_length: int) -> Dict[str, np.ndarray]:
        """Guarantee every required stem key exists."""
        for key in self.REQUIRED_STEMS:
            val = results.get(key)
            if val is None or (isinstance(val, np.ndarray) and val.size == 0):
                results[key] = np.zeros(reference_length, dtype=np.float32)
        return results

    def release(self) -> None:
        """
        FIX 2: Aggressively release Demucs + BS-Roformer from VRAM/RAM.
        Called after separation completes, before pitch phase starts.
        """
        if not self.config.release_memory_early:
            return
        try:
            import torch

            if self._demucs is not None:
                if hasattr(self._demucs, 'model') and self._demucs.model is not None:
                    del self._demucs.model
                self._demucs = None
                self._demucs_loaded = False

            if self._roformer is not None:
                if hasattr(self._roformer, 'model') and self._roformer.model is not None:
                    del self._roformer.model
                self._roformer = None
                self._roformer_loaded = False

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("   🧹 Separator VRAM cleared (GPU mode)")
            else:
                print("   🧹 Separator RAM freed (CPU mode)")
        except Exception as e:
            print(f"   ⚠️ release() error (non-fatal): {e}")

    async def _run_demucs_with_timeout(self, audio_path: Path, sr: int,
                                        duration: Optional[float]) -> Dict:
        """
        Run Demucs with timeout protection.

        FIX: The Demucs wrapper's separate() is called with `duration` so it
        calls librosa.load(..., duration=duration) internally rather than
        loading the whole file.  Check that your demucs.py DemucsSeparator
        passes duration to librosa.load — if not, see the companion fix there.
        """
        demucs = self._get_demucs()
        try:
            # Pass duration so Demucs only loads the truncated portion
            return await asyncio.wait_for(
                demucs.separate(audio_path, sr, duration=duration),
                timeout=self.config.demucs_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Demucs timed out after {self.config.demucs_timeout}s")

    async def _run_roformer_with_timeout(self, audio: np.ndarray,
                                          sr: int) -> Dict:
        """Run BS-Roformer with timeout protection."""
        roformer = self._get_roformer()
        if roformer is None:
            return {}
        try:
            return await asyncio.wait_for(
                roformer.separate(audio, sr),
                timeout=self.config.roformer_timeout
            )
        except asyncio.TimeoutError:
            print(f"   ⚠️ BS-Roformer timed out after {self.config.roformer_timeout}s")
            return {}
        except Exception as e:
            print(f"   ⚠️ BS-Roformer failed: {e}")
            return {}

    def _detect_artefacts(self, piano_stem: np.ndarray,
                           vocals_stem: np.ndarray,
                           sr: int) -> Tuple[bool, float]:
        if not self.config.enable_artefact_detection:
            return False, 0.0
        if piano_stem.size == 0 or vocals_stem.size == 0:
            return False, 0.0
        if np.mean(piano_stem ** 2) < 0.0001:
            return False, 0.0
        try:
            min_len = min(piano_stem.size, vocals_stem.size)
            piano_stem  = piano_stem[:min_len]
            vocals_stem = vocals_stem[:min_len]
            hop  = 512
            pc   = spectral_centroid(piano_stem, sr, hop_length=hop)
            vc   = spectral_centroid(vocals_stem, sr, hop_length=hop)
            mn   = min(len(pc), len(vc))
            if mn > 1:
                corr = float(np.corrcoef(pc[:mn], vc[:mn])[0, 1])
                if corr > 0.85:
                    print(f"   ⚠️ Artefact: piano centroid ~ vocals (r={corr:.2f})")
                return corr > 0.85, corr
        except Exception as e:
            print(f"   ⚠️ Artefact detection failed: {e}")
        return False, 0.0

    async def separate(self, audio_path: Path, sr: int,
                       duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems.

        Args:
            audio_path: Path to the original audio file
            sr:         Target sample rate
            duration:   FIX — Maximum seconds to process.  When provided,
                        neither Demucs nor any fallback librosa.load() will
                        read beyond this point.  Pass truncate_seconds from
                        the pipeline so Demucs never sees the full file.

        Returns dict with keys:
            drums, bass, vocals (from Demucs)
            piano, guitar, strings, winds, other_residual (from BS-Roformer)
        """
        start_time = time.time()

        is_valid, error_msg = self._validate_audio(audio_path)
        if not is_valid:
            raise ValueError(f"Audio validation failed: {error_msg}")

        self._update_progress(10, "Initializing separation models...")

        # ── STEP 1: Demucs ───────────────────────────────────────────────────
        print("🎛️ [1/2] Running Demucs separation"
              + (f" (first {duration:.0f}s)" if duration else "") + "...")
        self._update_progress(12, "Running Demucs stem separation...")

        try:
            # FIX: pass duration so Demucs only loads the truncated portion
            demucs_stems = await self._run_demucs_with_timeout(
                audio_path, sr, duration=duration)
            demucs_stems = self._ensure_demucs_stems(
                demucs_stems, audio_path, sr, duration=duration)
        except TimeoutError as e:
            print(f"   ❌ {e}")
            self._update_progress(20, "Demucs timeout — using fallback")
            # FIX: use duration, NOT hardcoded 60
            load_dur = duration if duration and duration > 0 else None
            y, _ = librosa.load(str(audio_path), sr=sr, duration=load_dur)
            return self._create_fallback_stems(y)
        except Exception as e:
            print(f"   ❌ Demucs failed: {e}")
            self._update_progress(20, "Demucs failed — using fallback")
            load_dur = duration if duration and duration > 0 else None
            y, _ = librosa.load(str(audio_path), sr=sr, duration=load_dur)
            return self._create_fallback_stems(y)

        results = {
            'drums':  demucs_stems.get('drums'),
            'bass':   demucs_stems.get('bass'),
            'vocals': demucs_stems.get('vocals'),
        }
        other_stem = demucs_stems.get('other')

        # ── FIX 2: Purge Demucs from memory BEFORE loading BS-Roformer ───────
        # This prevents OOM when both models try to occupy the same GPU/RAM.
        if self.config.release_memory_early:
            print("   🧹 Purging Demucs before BS-Roformer...")
            if self._demucs is not None:
                if hasattr(self._demucs, 'model') and self._demucs.model is not None:
                    del self._demucs.model
                    self._demucs.model = None
                self._demucs = None
                self._demucs_loaded = False
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("   🧹 Demucs VRAM cleared")
            except ImportError:
                pass

        has_other = (other_stem is not None and
                     isinstance(other_stem, np.ndarray) and
                     len(other_stem) > 0 and
                     np.mean(other_stem ** 2) > 0.0001)

        # ── STEP 2: BS-Roformer refines 'other' ──────────────────────────────
        self._update_progress(30, "Refining 'other' stem (BS-Roformer)...")

        if has_other and self.config.enable_roformer:
            # FIX: optionally truncate other_stem before passing to Roformer.
            # Even if Demucs produced a full-length stem, Roformer may only
            # need the first N seconds for template learning.
            roformer_audio = other_stem
            if self.config.truncate_for_roformer > 0:
                max_samples = self.config.truncate_for_roformer * sr
                if len(other_stem) > max_samples:
                    print(f"   ✂️ Truncating 'other' stem to "
                          f"{self.config.truncate_for_roformer}s for BS-Roformer "
                          f"(was {len(other_stem)/sr:.0f}s)")
                    roformer_audio = other_stem[:max_samples]

            print(f"🎯 [2/2] BS-Roformer on 'other' stem "
                  f"({len(roformer_audio)/sr:.0f}s)...")
            self._update_progress(35, "Running BS-Roformer...")

            bs_stems = await self._run_roformer_with_timeout(roformer_audio, sr)

            if bs_stems:
                ref = roformer_audio
                results['piano']          = bs_stems.get('piano',    np.zeros_like(ref))
                results['guitar']         = bs_stems.get('guitar',   np.zeros_like(ref))
                results['strings']        = bs_stems.get('strings',  np.zeros_like(ref))
                results['winds']          = bs_stems.get('winds',    np.zeros_like(ref))
                results['other_residual'] = bs_stems.get('other',    np.zeros_like(ref))

                self._update_progress(40, "Detecting separation artefacts...")
                # Numpy-safe fallback — never use `or` with numpy arrays because
                # array.__bool__() raises "truth value of array is ambiguous".
                _vocals_raw = results.get('vocals')
                vocals_stem = (
                    _vocals_raw
                    if (_vocals_raw is not None and
                        isinstance(_vocals_raw, np.ndarray) and
                        _vocals_raw.size > 0)
                    else np.zeros_like(ref)
                )
                is_artefact, corr = self._detect_artefacts(
                    results['piano'], vocals_stem, sr)
                results['piano_artefact_risk'] = is_artefact

                if is_artefact and self.state:
                    try:
                        self.state.add_evidence(
                            "separation_artefact", "piano_vocals_bleed",
                            corr, "hybrid_separator", start_time=0)
                    except Exception:
                        pass
            else:
                print("   ⚠️ BS-Roformer produced no stems — using 'other' as piano")
                results['piano']          = other_stem
                results['guitar']         = np.zeros_like(other_stem)
                results['strings']        = np.zeros_like(other_stem)
                results['winds']          = np.zeros_like(other_stem)
                results['other_residual'] = np.zeros_like(other_stem)
                results['piano_artefact_risk'] = False
        else:
            if not self.config.enable_roformer:
                print("   ⏭️ BS-Roformer disabled in config — using 'other' as piano")
            else:
                print("   ⏭️ 'Other' stem is silent — skipping BS-Roformer")

            silent = np.zeros(1, dtype=np.float32)
            if other_stem is not None and len(other_stem) > 0:
                results['piano'] = other_stem
            else:
                results['piano'] = silent
            results['guitar']         = silent
            results['strings']        = silent
            results['winds']          = silent
            results['other_residual'] = silent
            results['piano_artefact_risk'] = False

        # Guarantee all required keys exist
        reference_len = next(
            (len(results[s]) for s in ('drums', 'bass', 'vocals', 'piano')
             if results.get(s) is not None and len(results[s]) > 0),
            1
        )
        results = self._ensure_all_stems(results, reference_len)

        # Write stem energies to StateManager
        if self.state:
            try:
                energies = {k: float(np.mean(results[k] ** 2))
                            for k in self.REQUIRED_STEMS
                            if results.get(k) is not None and len(results[k]) > 0}
                self.state.add_evidence("stem_energies", energies, 0.8,
                                        "hybrid_separator", start_time=0)
            except Exception:
                pass

        elapsed = time.time() - start_time
        print(f"   ✅ Separation complete in {elapsed:.1f}s")
        self._update_progress(45, "Separation complete")
        return results


def create_hybrid_separator(use_streaming: bool = False,
                             demucs_timeout: float = 120.0,
                             roformer_timeout: float = 60.0,
                             truncate_for_roformer: int = 30,
                             state: Any = None,
                             progress_callback: Optional[Callable] = None
                             ) -> HybridSeparator:
    config = SeparationConfig(
        use_streaming=use_streaming,
        demucs_timeout=demucs_timeout,
        roformer_timeout=roformer_timeout,
        truncate_for_roformer=truncate_for_roformer,
    )
    return HybridSeparator(config=config, state=state,
                           progress_callback=progress_callback)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Hybrid Separator")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--truncate", type=int, default=60,
                        help="Max seconds to process (default 60)")
    args = parser.parse_args()

    async def test():
        print("=" * 60)
        print("🧪 Hybrid Separator Test")
        print("=" * 60)

        def on_progress(pct: int, msg: str):
            print(f"   [{pct:3d}%] {msg}")

        sep     = create_hybrid_separator(progress_callback=on_progress)
        results = await sep.separate(
            Path(args.audio_file), sr=44100,
            duration=float(args.truncate) if args.truncate > 0 else None
        )

        print("\n" + "=" * 60 + "\n📊 RESULTS\n" + "=" * 60)
        for key, stem in results.items():
            if isinstance(stem, np.ndarray):
                print(f"   {key:16s}: shape={stem.shape}, "
                      f"energy={np.mean(stem**2):.6f}")
            else:
                print(f"   {key:16s}: {stem}")

        sep.release()
        print("\n✅ Test complete")

    asyncio.run(test())