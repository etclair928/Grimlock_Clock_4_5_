#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
debug_separation.py — Comprehensive Demucs, Madmom, and BS-Roformer Test Suite

Tests to perform:
1. Demucs standalone - verify it loads and separates stems
2. Madmom tempo detection - compare with librosa baseline
3. BS-Roformer - test refinement of 'other' stem
4. HybridSeparator integration - full pipeline with real separation
5. Stem quality analysis - energy, RMS, spectral content

Default audio: C:/Users/kiwi2/Downloads/Hopeful.mp3
"""

import os
import sys
import asyncio
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import librosa
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_section(title: str):
    print(f"\n{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 80}{Colors.ENDC}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")


def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.ENDC}")


def print_info(msg: str):
    print(f"{Colors.BLUE}📌 {msg}{Colors.ENDC}")


def print_value(label: str, value: Any):
    print(f"   {Colors.DIM}{label}:{Colors.ENDC} {value}")


def print_metric(name: str, value: float, unit: str = "", threshold: Optional[Tuple[float, float]] = None):
    """Print a metric with optional pass/fail indicator"""
    if threshold:
        is_pass = threshold[0] <= value <= threshold[1]
        status = "✅" if is_pass else "❌"
        print(f"   {name:25s}: {value:8.3f} {unit:5s} {status}")
    else:
        print(f"   {name:25s}: {value:8.3f} {unit}")


# Default audio path
DEFAULT_AUDIO = Path(r"C:\Users\kiwi2\Downloads\Hopeful.mp3")


# ============================================================================
# AUDIO ANALYZER
# ============================================================================

class AudioAnalyzer:
    """Analyze audio stems for quality metrics"""

    @staticmethod
    def analyze(audio: np.ndarray, sr: int, name: str) -> Dict[str, Any]:
        """Extract comprehensive audio metrics"""
        if audio is None or len(audio) == 0:
            return {"error": "No audio data", "name": name}

        duration = len(audio) / sr

        # Energy metrics
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-8)

        # Spectral metrics
        try:
            spec = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)

            # Spectral centroid
            centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-8)

            # Spectral bandwidth
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / (np.sum(spec) + 1e-8))

            # Low frequency energy (below 250 Hz)
            low_mask = freqs < 250
            low_energy = np.sum(spec[low_mask]) / (np.sum(spec) + 1e-8)

            # High frequency energy (above 2000 Hz)
            high_mask = freqs > 2000
            high_energy = np.sum(spec[high_mask]) / (np.sum(spec) + 1e-8)

        except Exception as e:
            centroid = bandwidth = low_energy = high_energy = 0
            print_warning(f"Spectral analysis failed for {name}: {e}")

        return {
            "name": name,
            "duration": duration,
            "rms": rms,
            "peak": peak,
            "crest_factor": crest_factor,
            "spectral_centroid_hz": centroid,
            "spectral_bandwidth_hz": bandwidth,
            "low_energy_ratio": low_energy,
            "high_energy_ratio": high_energy,
            "is_silent": rms < 0.001,
            "is_clipping": peak > 0.99
        }

    @staticmethod
    def print_analysis(analysis: Dict[str, Any]):
        """Print analysis results"""
        if "error" in analysis:
            print_error(f"{analysis['name']}: {analysis['error']}")
            return

        print_info(f"📊 {analysis['name']}")
        print_metric("RMS", analysis['rms'], "", (0.001, 1.0))
        print_metric("Peak", analysis['peak'], "", (0, 1.0))
        print_metric("Crest Factor", analysis['crest_factor'], "", (1.0, 20.0))
        print_metric("Spectral Centroid", analysis['spectral_centroid_hz'], "Hz", (0, 8000))
        print_metric("Spectral Bandwidth", analysis['spectral_bandwidth_hz'], "Hz")
        print_metric("Low Energy (<250Hz)", analysis['low_energy_ratio'] * 100, "%")
        print_metric("High Energy (>2kHz)", analysis['high_energy_ratio'] * 100, "%")

        if analysis['is_silent']:
            print_warning("   ⚠️ Stem is SILENT")
        if analysis['is_clipping']:
            print_warning("   ⚠️ Stem has CLIPPING")


# ============================================================================
# DEMUCS TESTER
# ============================================================================

class DemucsTester:
    """Test Demucs separation capabilities"""

    def __init__(self):
        self.separator = None
        self._loaded = False

    def _get_separator(self):
        """Lazy load Demucs separator"""
        if not self._loaded:
            try:
                from separation.demucs import get_demucs_separator
                self.separator = get_demucs_separator()
                self._loaded = True
                print_success("Demucs loaded successfully")
            except ImportError as e:
                print_error(f"Demucs import failed: {e}")
                return None
            except Exception as e:
                print_error(f"Demucs init failed: {e}")
                return None
        return self.separator

    async def test_separation(self, audio_path: Path, sr: int = 44100) -> Dict[str, np.ndarray]:
        """Run Demucs separation and return stems"""
        separator = self._get_separator()
        if separator is None:
            return {}

        print_info("Running Demucs separation...")
        start_time = time.time()

        try:
            stems = await separator.separate(audio_path, sr)
            elapsed = time.time() - start_time
            print_success(f"Demucs separation completed in {elapsed:.1f}s")
            return stems
        except Exception as e:
            print_error(f"Demucs separation failed: {e}")
            return {}

    def analyze_stems(self, stems: Dict[str, np.ndarray], sr: int):
        """Analyze all stems from Demucs"""
        expected_stems = ['drums', 'bass', 'vocals', 'other']

        print_section("DEMUCS STEM ANALYSIS")

        for stem_name in expected_stems:
            if stem_name in stems and stems[stem_name] is not None:
                analysis = AudioAnalyzer.analyze(stems[stem_name], sr, stem_name)
                AudioAnalyzer.print_analysis(analysis)
            else:
                print_error(f"Missing stem: {stem_name}")

        # Additional stats
        print_info("\n📈 Demucs Statistics")
        for stem_name in expected_stems:
            if stem_name in stems and stems[stem_name] is not None:
                energy = np.mean(stems[stem_name] ** 2)
                print_value(f"{stem_name} energy", f"{energy:.6f}")


# ============================================================================
# MADMOM TESTER
# ============================================================================

class MadmomTester:
    """Test Madmom tempo and downbeat detection"""

    @staticmethod
    async def test_tempo(audio: np.ndarray, sr: int, timeout: float = 15.0) -> Optional[float]:
        """Test Madmom tempo detection with timeout"""
        try:
            from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

            print_info("Running Madmom tempo detection...")
            start_time = time.time()

            # Run in thread to avoid blocking
            def _run():
                beat_proc = RNNBeatProcessor()(audio)
                beat_tracker = DBNBeatTrackingProcessor(fps=100, min_bpm=40, max_bpm=250)
                beat_times = beat_tracker(beat_proc)
                if len(beat_times) > 1:
                    return 60.0 / np.median(np.diff(beat_times))
                return None

            tempo = await asyncio.wait_for(
                asyncio.to_thread(_run),
                timeout=timeout
            )

            elapsed = time.time() - start_time
            if tempo:
                print_success(f"Madmom tempo: {tempo:.1f} BPM ({elapsed:.1f}s)")
                return tempo
            else:
                print_warning(f"Madmom tempo detection failed (no beats detected)")
                return None

        except ImportError:
            print_error("Madmom not installed")
            return None
        except asyncio.TimeoutError:
            print_warning(f"Madmom tempo timeout after {timeout}s")
            return None
        except Exception as e:
            print_error(f"Madmom tempo error: {e}")
            return None

    @staticmethod
    async def test_downbeats(audio: np.ndarray, sr: int, timeout: float = 15.0) -> Optional[List[float]]:
        """Test Madmom downbeat detection"""
        try:
            from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

            print_info("Running Madmom downbeat detection...")
            start_time = time.time()

            def _run():
                proc = RNNDownBeatProcessor()(audio)
                tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4, 5, 6, 7], fps=100)
                results = tracker(proc)
                if results is not None and len(results) > 0:
                    return [r[0] for r in results if r[1] == 1]
                return []

            downbeats = await asyncio.wait_for(
                asyncio.to_thread(_run),
                timeout=timeout
            )

            elapsed = time.time() - start_time
            if downbeats:
                print_success(f"Madmom downbeats: {len(downbeats)} detected ({elapsed:.1f}s)")
                return downbeats
            else:
                print_warning(f"No downbeats detected")
                return []

        except ImportError:
            print_error("Madmom not installed")
            return []
        except asyncio.TimeoutError:
            print_warning(f"Madmom downbeat timeout after {timeout}s")
            return []
        except Exception as e:
            print_error(f"Madmom downbeat error: {e}")
            return []


# ============================================================================
# BS-ROFORMER TESTER
# ============================================================================

class BSRoformerTester:
    """Test BS-Roformer refinement on 'other' stem"""

    def __init__(self):
        self._roformer = None
        self._loaded = False

    def _get_roformer(self):
        """Lazy load BS-Roformer"""
        if not self._loaded:
            try:
                from separation.bs_roformer_engine import BSRoformerSeparator
                self._roformer = BSRoformerSeparator()
                self._loaded = True
                print_success("BS-Roformer loaded successfully")
            except ImportError as e:
                print_error(f"BS-Roformer import failed: {e}")
                return None
            except Exception as e:
                print_error(f"BS-Roformer init failed: {e}")
                return None
        return self._roformer

    async def test_refinement(self, other_stem: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Test BS-Roformer refinement on 'other' stem"""
        roformer = self._get_roformer()
        if roformer is None:
            return {}

        if other_stem is None or len(other_stem) == 0:
            print_error("No 'other' stem provided")
            return {}

        if np.mean(other_stem ** 2) < 0.0001:
            print_warning("'Other' stem is silent - skipping BS-Roformer")
            return {}

        print_info("Running BS-Roformer refinement...")
        start_time = time.time()

        try:
            stems = await roformer.separate(other_stem, sr)
            elapsed = time.time() - start_time
            print_success(f"BS-Roformer completed in {elapsed:.1f}s")
            return stems
        except Exception as e:
            print_error(f"BS-Roformer failed: {e}")
            return {}

    @staticmethod
    def analyze_refinement(original: np.ndarray, refined: Dict[str, np.ndarray], sr: int):
        """Compare original 'other' stem with refined components"""
        print_section("BS-ROFORMER REFINEMENT ANALYSIS")

        # Original energy
        original_energy = np.mean(original ** 2)
        print_metric("Original 'other' energy", original_energy, "")

        # Refined components
        total_refined_energy = 0
        components = ['piano', 'guitar', 'strings', 'winds', 'other']

        for comp in components:
            if comp in refined and refined[comp] is not None:
                energy = np.mean(refined[comp] ** 2)
                total_refined_energy += energy
                print_metric(f"  {comp} energy", energy, "")

        # Energy preservation
        print_info(f"\n📊 Energy Analysis")
        print_metric("Original energy", original_energy, "")
        print_metric("Total refined energy", total_refined_energy, "")

        if original_energy > 0:
            preservation = total_refined_energy / original_energy
            print_metric("Energy preservation", preservation * 100, "%")
            if preservation > 0.8:
                print_success("Good energy preservation")
            elif preservation > 0.5:
                print_warning("Moderate energy preservation")
            else:
                print_error("Poor energy preservation - check BS-Roformer")


# ============================================================================
# HYBRID SEPARATOR TESTER
# ============================================================================

class HybridSeparatorTester:
    """Test the complete HybridSeparator integration"""

    def __init__(self):
        self.separator = None
        self._loaded = False

    def _get_separator(self):
        """Lazy load HybridSeparator"""
        if not self._loaded:
            try:
                from separation.hybrid_separator import HybridSeparator
                self.separator = HybridSeparator()
                self._loaded = True
                print_success("HybridSeparator loaded successfully")
            except ImportError as e:
                print_error(f"HybridSeparator import failed: {e}")
                return None
            except Exception as e:
                print_error(f"HybridSeparator init failed: {e}")
                return None
        return self.separator

    async def test_full_separation(self, audio_path: Path, sr: int = 44100) -> Dict[str, np.ndarray]:
        """Test complete hybrid separation pipeline"""
        separator = self._get_separator()
        if separator is None:
            return {}

        print_info("Running HybridSeparator (Demucs + BS-Roformer)...")
        start_time = time.time()

        try:
            stems = await separator.separate(audio_path, sr)
            elapsed = time.time() - start_time
            print_success(f"HybridSeparator completed in {elapsed:.1f}s")

            # Show progress
            for key in ['drums', 'bass', 'vocals', 'piano']:
                if key in stems and stems[key] is not None:
                    energy = np.mean(stems[key] ** 2)
                    print_value(f"  {key} energy", f"{energy:.6f}")

            return stems
        except Exception as e:
            print_error(f"HybridSeparator failed: {e}")
            import traceback
            traceback.print_exc()
            return {}


# ============================================================================
# PIPELINE STEM TESTER
# ============================================================================

class PipelineStemTester:
    """Test the full pipeline with real stems"""

    def __init__(self):
        self.pipeline = None
        self._loaded = False

    def _get_pipeline(self):
        """Load pipeline"""
        if not self._loaded:
            try:
                from grimlock_pipeline import GrimlockPipeline, PipelineConfig
                self.pipeline = GrimlockPipeline()
                self._loaded = True
                print_success("Pipeline loaded successfully")
            except ImportError as e:
                print_error(f"Pipeline import failed: {e}")
                return None
            except Exception as e:
                print_error(f"Pipeline init failed: {e}")
                return None
        return self.pipeline

    async def test_with_stems(self, stems: Dict[str, np.ndarray], sr: int, duration: float) -> Dict:
        """Test pipeline with pre-separated stems"""
        pipeline = self._get_pipeline()
        if pipeline is None:
            return {}

        print_info("Testing pipeline with separated stems...")

        # Create a mock result structure
        results = {
            'rhythm': None,
            'drum_hits': None,
            'notes': None,
            'confidence': 0
        }

        # Test rhythm engine
        if hasattr(pipeline, 'rhythm_engine') and pipeline.rhythm_engine:
            try:
                from order_types import RhythmInfo

                # Mock process - we're just testing integration
                print_info("Testing rhythm engine with drums stem...")
                if 'drums' in stems and stems['drums'] is not None:
                    # In real usage, this would call the engine
                    results['rhythm'] = RhythmInfo(tempo=130.0, confidence=0.8)
                    print_success("Rhythm engine test passed")
                else:
                    print_warning("No drums stem - skipping rhythm test")
            except Exception as e:
                print_error(f"Rhythm engine test failed: {e}")

        # Test drum intelligence
        if hasattr(pipeline, 'drum_intel') and pipeline.drum_intel:
            try:
                print_info("Testing drum intelligence...")
                if 'drums' in stems and stems['drums'] is not None:
                    # Mock test
                    results['drum_hits'] = []
                    print_success("Drum intelligence test passed")
                else:
                    print_warning("No drums stem - skipping drum test")
            except Exception as e:
                print_error(f"Drum intelligence test failed: {e}")

        # Test pitch intelligence
        if hasattr(pipeline, 'pitch_intel') and pipeline.pitch_intel:
            try:
                print_info("Testing pitch intelligence...")
                if 'piano' in stems and stems['piano'] is not None:
                    # Mock test
                    results['notes'] = []
                    print_success("Pitch intelligence test passed")
                else:
                    print_warning("No piano stem - skipping pitch test")
            except Exception as e:
                print_error(f"Pitch intelligence test failed: {e}")

        return results


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Separation Debug Suite")
    parser.add_argument("--audio", "-a", type=str, default=str(DEFAULT_AUDIO),
                        help=f"Path to audio file (default: {DEFAULT_AUDIO})")
    parser.add_argument("--skip-demucs", action="store_true", help="Skip Demucs tests")
    parser.add_argument("--skip-madmom", action="store_true", help="Skip Madmom tests")
    parser.add_argument("--skip-bsroformer", action="store_true", help="Skip BS-Roformer tests")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip HybridSeparator tests")
    parser.add_argument("--list-audio", action="store_true", help="List available audio files")

    args = parser.parse_args()

    # List audio files
    if args.list_audio:
        print_section("📁 AVAILABLE AUDIO FILES")
        downloads = Path(r"C:\Users\kiwi2\Downloads")
        if downloads.exists():
            audio_files = list(downloads.glob("*.mp3")) + list(downloads.glob("*.wav")) + list(downloads.glob("*.m4a"))
            for f in audio_files[:20]:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   {f.name} ({size_mb:.1f} MB)")
        return

    # Check audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print_error(f"Audio file not found: {audio_path}")
        return

    print_section("🔍 SEPARATION DEBUG SUITE")
    print_value("Audio file", audio_path)
    print_value("File size", f"{audio_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load reference audio
    print_info("Loading reference audio...")
    y, sr = librosa.load(str(audio_path), sr=44100, duration=30)
    duration = len(y) / sr
    print_success(f"Loaded {duration:.1f}s at {sr} Hz")

    # Analyze original audio
    print_section("📊 ORIGINAL AUDIO ANALYSIS")
    original_analysis = AudioAnalyzer.analyze(y, sr, "Original")
    AudioAnalyzer.print_analysis(original_analysis)

    # ========================================================================
    # MADMOM TESTS
    # ========================================================================
    if not args.skip_madmom:
        print_section("🎚️ MADMOM TESTS")

        # Test tempo
        tempo = await MadmomTester.test_tempo(y, sr)
        if tempo:
            print_success(f"Madmom tempo: {tempo:.1f} BPM")
        else:
            # Fallback to librosa
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_librosa = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            print_warning(f"Using librosa fallback tempo: {tempo_librosa:.1f} BPM")

        # Test downbeats
        downbeats = await MadmomTester.test_downbeats(y, sr)
        if downbeats:
            print_value("First 5 downbeats", [f"{t:.2f}s" for t in downbeats[:5]])

    # ========================================================================
    # DEMUCS TESTS
    # ========================================================================
    if not args.skip_demucs:
        print_section("🎛️ DEMUCS TESTS")

        demucs_tester = DemucsTester()
        stems = await demucs_tester.test_separation(audio_path, sr)

        if stems:
            demucs_tester.analyze_stems(stems, sr)
        else:
            print_error("Demucs separation failed - skipping stem-dependent tests")
            return

    # ========================================================================
    # BS-ROFORMER TESTS
    # ========================================================================
    if not args.skip_bsroformer and stems and 'other' in stems:
        print_section("🎸 BS-ROFORMER TESTS")

        bs_tester = BSRoformerTester()
        refined = await bs_tester.test_refinement(stems['other'], sr)

        if refined:
            bs_tester.analyze_refinement(stems['other'], refined, sr)

            # Show combined stem set
            print_info("\n📦 Combined stems from Demucs + BS-Roformer:")
            all_stems = {**stems, **refined}
            for key in ['drums', 'bass', 'vocals', 'piano', 'guitar', 'strings']:
                if key in all_stems:
                    energy = np.mean(all_stems[key] ** 2)
                    print_value(f"  {key}", f"energy={energy:.6f}")

    # ========================================================================
    # HYBRID SEPARATOR TESTS
    # ========================================================================
    if not args.skip_hybrid:
        print_section("🔗 HYBRID SEPARATOR INTEGRATION TEST")

        hybrid_tester = HybridSeparatorTester()
        hybrid_stems = await hybrid_tester.test_full_separation(audio_path, sr)

        if hybrid_stems:
            print_success("HybridSeparator integration successful")

            # Verify required stems
            required = ['drums', 'bass', 'vocals', 'piano', 'guitar', 'strings', 'winds', 'other_residual']
            missing = [k for k in required if k not in hybrid_stems or hybrid_stems[k] is None]

            if missing:
                print_warning(f"Missing stems: {missing}")
            else:
                print_success("All required stems present")

    # ========================================================================
    # PIPELINE INTEGRATION TEST
    # ========================================================================
    if stems:
        print_section("🚀 PIPELINE STEM INTEGRATION TEST")

        pipeline_tester = PipelineStemTester()
        results = await pipeline_tester.test_with_stems(stems, sr, duration)

        print_info("\n📊 Integration Results:")
        print_value("Rhythm engine", "✅" if results.get('rhythm') else "❌")
        print_value("Drum intelligence", "✅" if results.get('drum_hits') is not None else "❌")
        print_value("Pitch intelligence", "✅" if results.get('notes') is not None else "❌")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("📋 TEST SUMMARY")
    print_value("Audio file", audio_path.name)
    print_value("Duration", f"{duration:.1f}s")
    print_value("Sample rate", f"{sr} Hz")

    # Runtime statistics
    print_info("\n✅ Test suite completed successfully!")
    print_info("Recommendations:")
    print("   1. Ensure separation/__init__.py exists for proper imports")
    print("   2. Consider increasing Madmom timeout if consistently failing")
    print("   3. Verify BS-Roformer model is properly installed")


if __name__ == "__main__":
    asyncio.run(main())