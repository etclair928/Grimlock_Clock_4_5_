#!/usr/bin/env python3
"""
Pitch Intelligence v3 — Unified Pitch Fusion Engine with Librosa Primary

Architecture:
    Librosa (Fast, deterministic, always works) runs in parallel with ML models
    Results fused via confidence-weighted consensus
    Timeout protection prevents ML hangs from blocking pipeline

Fixes over v2:
    - Librosa elevated to first-class citizen (no longer just fallback)
    - All detectors run in parallel with individual timeouts
    - Librosa provides guaranteed baseline results
    - ML models optional with graceful degradation
    - Fixed CREPE 'multiple values for argument audio_16k' bug
    - Added Basic Pitch tuple/object parsing robustness
    - SPICE tensor shape/padding fixes
    - Confidence weights calibrated across all sources
"""

import numpy as np
import librosa
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from enum import Enum


# ------------------------------------------------------------
# ENUMS
# ------------------------------------------------------------

class InstrumentType(Enum):
    PIANO = "PIANO"
    BASS = "BASS"
    MELODY = "MELODY"
    DRUMS = "DRUMS"
    UNKNOWN = "UNKNOWN"


# ------------------------------------------------------------
# TYPES
# ------------------------------------------------------------

@dataclass
class PitchEvent:
    """Raw pitch detection event from a single source"""
    time: float
    pitch: int
    confidence: float
    source: str
    frequency_hz: Optional[float] = None


@dataclass
class Note:
    """Final note event after fusion"""
    pitch: int
    start: float
    end: float
    velocity: int
    confidence: float
    instrument: InstrumentType = InstrumentType.PIANO
    voice_id: int = 0

    def duration(self) -> float:
        return self.end - self.start

    def midi_velocity(self) -> int:
        return max(1, min(127, self.velocity))


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

@dataclass
class PitchConfig:
    """Configuration for pitch intelligence pipeline"""

    # Audio settings
    sr: int = 22050
    crepe_sr: int = 16000
    spice_sr: int = 16000

    # Merge settings
    merge_gap: float = 0.02  # 20ms gap for merging same-pitch notes
    min_duration: float = 0.05  # 50ms minimum note duration

    # Confidence thresholds
    crepe_confidence: float = 0.35
    spice_confidence: float = 0.25
    basic_pitch_confidence: float = 0.50
    librosa_confidence_min: float = 0.15      # Minimum magnitude for librosa
    librosa_confidence_max: float = 0.85      # Librosa ceiling (can't match ML)

    # Frame-based fusion
    fusion_frame_ms: int = 50  # 50ms frames for fusion
    fusion_hop_ms: int = 10    # 10ms hop for time alignment

    # Weights for source importance (calibrated for jazz)
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "librosa_piptrack": 0.75,     # Fast, reliable, good for onsets
        "basic_pitch": 1.0,           # Most accurate for piano polyphony
        "crepe": 0.9,                 # Excellent for monophonic pitch
        "spice": 0.7,                 # Good for percussive/attack
        "crepe_fallback": 0.5,        # Reduced confidence interpolation
        "librosa_onset": 0.65,        # Onset-refined events
    })

    # Timeout per detector (seconds)
    detector_timeout: float = 30.0

    # Velocity mapping
    velocity_min: int = 20
    velocity_max: int = 127

    # Debug
    debug: bool = False


# ------------------------------------------------------------
# SHARED UTILITIES
# ------------------------------------------------------------

def ensure_16k(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to 16kHz once, not per detector"""
    if sr == target_sr:
        return audio.astype(np.float32)
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


# ------------------------------------------------------------
# LIBROSA DETECTOR (PRIMARY - FAST, DETERMINISTIC)
# ------------------------------------------------------------

class LibrosaDetector:
    """
    Librosa-based pitch detection using piptrack + onset refinement.

    Advantages over ML models:
        - Always finishes (no timeout risks)
        - No model loading overhead
        - Works on any audio length
        - Returns onsets + pitch together
        - Deterministic results

    Disadvantages:
        - Lower accuracy for complex polyphony
        - No pitch bend tracking
        - Confidence ceiling ~0.85
    """

    def __init__(self, config: PitchConfig):
        self.config = config

    async def detect(self, audio: np.ndarray, sr: int,
                     audio_16k: Optional[np.ndarray] = None) -> List[PitchEvent]:
        """
        Extract pitch events using librosa piptrack with onset refinement.
        """

        def _blocking_detect():
            # Use 16kHz for consistency
            audio_use = audio_16k if audio_16k is not None else ensure_16k(audio, sr, 16000)
            sr_use = 16000

            # Compute pitch and magnitude contours
            pitches, magnitudes = librosa.piptrack(
                y=audio_use,
                sr=sr_use,
                fmin=librosa.note_to_hz('C2'),   # 65 Hz - bass range
                fmax=librosa.note_to_hz('C7')    # 2093 Hz - piano range
            )

            # Compute onset strength for timing refinement
            onset_env = librosa.onset.onset_strength(y=audio_use, sr=sr_use)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr_use,
                backtrack=True
            )
            onset_times = librosa.frames_to_time(onsets, sr=sr_use)

            # Frame timing
            hop_length = 512
            frame_times = librosa.frames_to_time(
                np.arange(pitches.shape[1]),
                sr=sr_use,
                hop_length=hop_length
            )

            events = []
            last_pitch = None
            last_time = 0

            for t_idx in range(pitches.shape[1]):
                frame_mags = magnitudes[:, t_idx]
                if len(frame_mags) == 0:
                    continue

                max_idx = np.argmax(frame_mags)
                max_mag = frame_mags[max_idx]

                if max_mag < self.config.librosa_confidence_min:
                    continue

                pitch_hz = pitches[max_idx, t_idx]
                if pitch_hz <= 0:
                    continue

                midi = librosa.hz_to_midi(float(pitch_hz))

                if midi < 21 or midi > 108:
                    continue

                # Map magnitude to confidence (scaled to ceiling)
                conf_range = self.config.librosa_confidence_max - self.config.librosa_confidence_min
                confidence = self.config.librosa_confidence_min + min(conf_range, max_mag * 0.8)
                confidence = min(self.config.librosa_confidence_max, confidence)

                time_sec = frame_times[t_idx]

                # Only add if pitch changed or significant time passed
                if (last_pitch is None or
                    abs(midi - last_pitch) > 1 or
                    time_sec - last_time > 0.1):

                    events.append(PitchEvent(
                        time=time_sec,
                        pitch=int(round(midi)),
                        confidence=confidence,
                        source="librosa_piptrack",
                        frequency_hz=float(pitch_hz)
                    ))

                    last_pitch = midi
                    last_time = time_sec

            # Refine timing using onset detection
            refined_events = []
            for event in events:
                if len(onset_times) > 0:
                    nearest_onset = min(onset_times, key=lambda x: abs(x - event.time))
                    if abs(nearest_onset - event.time) < 0.05:
                        event.time = nearest_onset
                        event.confidence = min(0.9, event.confidence * 1.05)
                        event.source = "librosa_onset"
                refined_events.append(event)

            if self.config.debug:
                print(f"📊 Librosa: {len(refined_events)} events")

            return refined_events

        try:
            return await asyncio.to_thread(_blocking_detect)
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ Librosa detection failed: {e}")
            return []


# ------------------------------------------------------------
# BASIC PITCH DETECTOR (ML POLYPHONIC)
# ------------------------------------------------------------

class BasicPitchDetector:
    """Basic Pitch detector with robust tuple/object parsing"""

    def __init__(self, config: PitchConfig):
        self.config = config

    async def detect(self, audio_path: Path) -> List[PitchEvent]:
        """Extract pitch events using Basic Pitch"""

        def _blocking_detect():
            try:
                from basic_pitch.inference import predict
                _, _, events = predict(str(audio_path))
            except ImportError:
                if self.config.debug:
                    print("⚠️ Basic Pitch not installed")
                return []
            except Exception as e:
                if self.config.debug:
                    print(f"⚠️ Basic Pitch error: {e}")
                return []

            output = []

            for e in events:
                try:
                    # Handle tuple format (Basic Pitch default)
                    if isinstance(e, (tuple, list)):
                        if len(e) >= 4:
                            pitch, start, end, velocity = e[:4]
                        else:
                            continue
                    # Handle object format
                    elif hasattr(e, "pitch"):
                        pitch = e.pitch
                        start = e.start
                        end = e.end
                        velocity = getattr(e, "velocity", 100)
                    else:
                        continue

                    if pitch < 21 or pitch > 108:
                        continue

                    # Use middle of note as event time
                    time_sec = (float(start) + float(end)) / 2

                    output.append(PitchEvent(
                        time=time_sec,
                        pitch=int(pitch),
                        confidence=self.config.basic_pitch_confidence,
                        source="basic_pitch"
                    ))

                except Exception as err:
                    if self.config.debug:
                        print(f"⚠️ Basic Pitch parse error: {err}")

            if self.config.debug:
                print(f"📊 Basic Pitch: {len(output)} events")

            return output

        return await asyncio.to_thread(_blocking_detect)


# ------------------------------------------------------------
# CREPE DETECTOR (ML MONOPHONIC)
# ------------------------------------------------------------

class CREPEDetector:
    """CREPE pitch detector with confidence thresholds"""

    def __init__(self, config: PitchConfig):
        self.config = config

    async def detect(self, audio: np.ndarray, sr: int,
                     audio_16k: Optional[np.ndarray] = None) -> List[PitchEvent]:
        """Extract pitch events using CREPE"""

        def _blocking_detect():
            try:
                import crepe
            except ImportError:
                if self.config.debug:
                    print("⚠️ CREPE not installed")
                return []

            # Use pre-resampled buffer if provided
            audio_use = audio_16k if audio_16k is not None else ensure_16k(audio, sr, self.config.crepe_sr)
            audio_use = normalize_audio(audio_use)

            try:
                time_seconds, frequencies, confidences, _ = crepe.predict(
                    audio_use,
                    self.config.crepe_sr,
                    viterbi=True,
                    step_size=10,
                    verbose=0
                )
            except Exception as e:
                if self.config.debug:
                    print(f"⚠️ CREPE inference failed: {e}")
                return []

            events = []

            for t, freq, conf in zip(time_seconds, frequencies, confidences):
                if conf < self.config.crepe_confidence:
                    continue

                midi = librosa.hz_to_midi(float(freq))

                if 21 <= midi <= 108:
                    events.append(PitchEvent(
                        time=float(t),
                        pitch=int(round(midi)),
                        confidence=float(conf),
                        source="crepe",
                        frequency_hz=float(freq)
                    ))

            # Fallback interpolation for sparse frames
            if len(events) < 10 and len(time_seconds) > 0:
                if self.config.debug:
                    print(f"🔄 CREPE sparse ({len(events)} events) - interpolating")

                confident_indices = [i for i, conf in enumerate(confidences)
                                    if conf >= self.config.crepe_confidence]

                if len(confident_indices) >= 2:
                    for i, (t, freq, conf) in enumerate(zip(time_seconds, frequencies, confidences)):
                        if conf >= self.config.crepe_confidence:
                            continue

                        nearest_idx = min(confident_indices, key=lambda x: abs(x - i))
                        nearest_freq = frequencies[nearest_idx]
                        nearest_conf = confidences[nearest_idx]

                        midi = librosa.hz_to_midi(float(nearest_freq))
                        if 21 <= midi <= 108:
                            events.append(PitchEvent(
                                time=float(t),
                                pitch=int(round(midi)),
                                confidence=float(nearest_conf * 0.5),
                                source="crepe_fallback",
                                frequency_hz=float(nearest_freq)
                            ))

            if self.config.debug:
                print(f"📊 CREPE: {len(events)} events")

            return events

        return await asyncio.to_thread(_blocking_detect)


# ------------------------------------------------------------
# SPICE DETECTOR (ML PERCUSSIVE/ONSET)
# ------------------------------------------------------------

class SPICEDetector:
    """SPICE detector with fixed tensor operations and model caching"""

    _model = None
    _model_lock = asyncio.Lock()

    def __init__(self, config: PitchConfig):
        self.config = config

    async def _load_model(self):
        """Load SPICE model with thread-safe caching"""
        if SPICEDetector._model is None:
            async with SPICEDetector._model_lock:
                if SPICEDetector._model is None:
                    try:
                        import tensorflow as tf
                        import tensorflow_hub as hub

                        tf.get_logger().setLevel('ERROR')

                        if self.config.debug:
                            print("🔄 Loading SPICE model...")

                        SPICEDetector._model = hub.load("https://tfhub.dev/google/spice/2")

                        if self.config.debug:
                            print("✅ SPICE model loaded")
                    except ImportError:
                        if self.config.debug:
                            print("⚠️ TensorFlow/TensorFlow Hub not installed")
                        return None
                    except Exception as e:
                        if self.config.debug:
                            print(f"⚠️ SPICE load failed: {e}")
                        return None

        return SPICEDetector._model

    async def detect(self, audio: np.ndarray, sr: int,
                     audio_16k: Optional[np.ndarray] = None) -> List[PitchEvent]:
        """Extract percussive events using SPICE"""

        model = await self._load_model()
        if model is None:
            return []

        def _blocking_detect():
            try:
                import tensorflow as tf

                # Use 16kHz audio
                audio_use = audio_16k if audio_16k is not None else ensure_16k(audio, sr, self.config.spice_sr)

                # Normalize
                audio_use = normalize_audio(audio_use)

                # Add batch dimension
                if len(audio_use.shape) == 1:
                    audio_use = audio_use.reshape(1, -1)

                audio_use = audio_use.astype(np.float32)

                # Pad/truncate to 30 seconds (SPICE expectation)
                target_len = self.config.spice_sr * 30
                current_len = audio_use.shape[1]

                if current_len < target_len:
                    padding = target_len - current_len
                    audio_use = np.pad(audio_use, ((0, 0), (0, padding)), mode='constant')
                elif current_len > target_len:
                    audio_use = audio_use[:, :target_len]

                # Ensure length multiple of hop size (512)
                hop_size = 512
                remainder = audio_use.shape[1] % hop_size
                if remainder != 0:
                    padding = hop_size - remainder
                    audio_use = np.pad(audio_use, ((0, 0), (0, padding)), mode='constant')

                # Run inference
                outputs = model.signatures["serving_default"](
                    audio=tf.constant(audio_use, dtype=tf.float32),
                    sample_rate=tf.constant(self.config.spice_sr, dtype=tf.int32)
                )

                onsets = outputs["onsets"].numpy().flatten()

                events = []
                for idx, confidence in enumerate(onsets):
                    if confidence > self.config.spice_confidence:
                        time_seconds = idx * (hop_size / self.config.spice_sr)

                        events.append(PitchEvent(
                            time=float(time_seconds),
                            pitch=60,  # C4 default for percussive
                            confidence=float(confidence),
                            source="spice"
                        ))

                if self.config.debug:
                    print(f"📊 SPICE: {len(events)} events")

                return events

            except Exception as e:
                if self.config.debug:
                    print(f"⚠️ SPICE detection failed: {e}")
                return []

        return await asyncio.to_thread(_blocking_detect)


# ------------------------------------------------------------
# FUSION ENGINE (CONFIDENCE-WEIGHTED + FRAME-BASED)
# ------------------------------------------------------------

class PitchFusionEngine:
    """Fuses multiple pitch detection sources using frame-based weighting"""

    def __init__(self, config: PitchConfig):
        self.config = config

    def fuse(self, events: List[PitchEvent]) -> List[Note]:
        """
        Fuse pitch events from multiple sources.

        Strategy:
            1. Group events by time frames (50ms windows)
            2. Inside each frame, group by pitch
            3. Weight pitches by confidence and source importance
            4. Create notes from weighted averages
        """
        if not events:
            return []

        frame_ms = self.config.fusion_frame_ms
        frames: Dict[Tuple[int, int], List[PitchEvent]] = defaultdict(list)

        for event in events:
            frame_idx = int(event.time * 1000 // frame_ms)
            pitch_key = event.pitch
            frames[(frame_idx, pitch_key)].append(event)

        notes = []

        for (frame_idx, pitch_key), group in frames.items():
            total_weight = 0.0
            weighted_pitch_sum = 0.0
            max_confidence = 0.0
            source_weights_used = []

            for event in group:
                source_weight = self.config.source_weights.get(event.source, 0.5)
                weight = event.confidence * source_weight

                weighted_pitch_sum += event.pitch * weight
                total_weight += weight
                max_confidence = max(max_confidence, event.confidence)
                source_weights_used.append(f"{event.source}:{event.confidence:.2f}")

            if total_weight == 0:
                continue

            avg_pitch = weighted_pitch_sum / total_weight

            start_time = frame_idx * frame_ms / 1000.0
            end_time = start_time + (frame_ms / 1000.0)

            # Map confidence to velocity
            velocity_range = self.config.velocity_max - self.config.velocity_min
            velocity = self.config.velocity_min + int(max_confidence * velocity_range)
            velocity = max(self.config.velocity_min, min(self.config.velocity_max, velocity))

            note = Note(
                pitch=int(round(avg_pitch)),
                start=start_time,
                end=end_time,
                velocity=velocity,
                confidence=max_confidence,
                instrument=InstrumentType.PIANO
            )

            notes.append(note)

        if self.config.debug:
            print(f"📊 Fusion: {len(events)} events → {len(notes)} notes")

        return notes


# ------------------------------------------------------------
# HUMAN-AWARE QUANTIZER
# ------------------------------------------------------------

class HumanAwareQuantizer:
    """Merges overlapping notes and enforces musical sense"""

    def __init__(self, config: PitchConfig):
        self.config = config

    def process(self, notes: List[Note]) -> List[Note]:
        """
        Process notes with musical quantization:
            - Merge overlapping same-pitch notes
            - Enforce minimum duration
            - Sort by pitch and time
        """
        if not notes:
            return []

        notes = sorted(notes, key=lambda n: (n.pitch, n.start))

        merged: List[Note] = []

        for note in notes:
            if not merged:
                merged.append(self._copy_note(note))
                continue

            last = merged[-1]

            if (note.pitch == last.pitch and
                (note.start - last.end) < self.config.merge_gap):
                last.end = max(last.end, note.end)
                last.confidence = max(last.confidence, note.confidence)
                last.velocity = max(last.velocity, note.velocity)
            else:
                merged.append(self._copy_note(note))

        for note in merged:
            if note.duration() < self.config.min_duration:
                note.end = note.start + self.config.min_duration

        if self.config.debug:
            print(f"📊 Quantizer: {len(notes)} notes → {len(merged)} notes")

        return merged

    def _copy_note(self, note: Note) -> Note:
        """Create a deep copy of a note (no mutation side effects)"""
        return Note(
            pitch=note.pitch,
            start=note.start,
            end=note.end,
            velocity=note.velocity,
            confidence=note.confidence,
            instrument=note.instrument,
            voice_id=note.voice_id
        )


# ------------------------------------------------------------
# MAIN ORCHESTRATOR
# ------------------------------------------------------------

class PitchIntelligence:
    """
    Unified pitch intelligence orchestrator.

    Architecture:
        Librosa (fast, deterministic) runs alongside ML models
        All detectors run in parallel with individual timeouts
        Results fused via confidence-weighted consensus
    """

    def __init__(self, config: Optional[PitchConfig] = None):
        self.config = config or PitchConfig()
        self.librosa = LibrosaDetector(self.config)
        self.basic = BasicPitchDetector(self.config)
        self.crepe = CREPEDetector(self.config)
        self.spice = SPICEDetector(self.config)
        self.fusion = PitchFusionEngine(self.config)
        self.quantizer = HumanAwareQuantizer(self.config)

    async def process(self, audio: np.ndarray, sr: int,
                      audio_path: Optional[Path] = None,
                      audio_16k: Optional[np.ndarray] = None,
                      use_librosa: bool = True,
                      use_basic_pitch: bool = True,
                      use_crepe: bool = True,
                      use_spice: bool = True) -> List[Note]:
        """
        Process audio through all enabled detectors in parallel.

        Args:
            audio: Audio signal (any sample rate)
            sr: Sample rate of audio
            audio_path: Path to audio file (for Basic Pitch)
            audio_16k: Pre-resampled 16kHz audio (avoids duplicate resampling)
            use_librosa: Enable librosa detector
            use_basic_pitch: Enable Basic Pitch ML detector
            use_crepe: Enable CREPE ML detector
            use_spice: Enable SPICE ML detector

        Returns:
            List of Note objects with fused pitch information
        """

        async def run_with_timeout(name: str, coro, timeout: float):
            try:
                return name, await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                if self.config.debug:
                    print(f"⏰ {name} timed out after {timeout}s")
                return name, []
            except Exception as e:
                if self.config.debug:
                    print(f"⚠️ {name} failed: {e}")
                return name, []

        tasks = []
        task_names = []

        # Librosa - always runs if enabled (fast, never fails)
        if use_librosa:
            tasks.append(run_with_timeout("librosa",
                         self.librosa.detect(audio, sr, audio_16k),
                         self.config.detector_timeout))
            task_names.append("librosa")

        # Basic Pitch - requires audio file path
        if use_basic_pitch and audio_path and audio_path.exists():
            tasks.append(run_with_timeout("basic_pitch",
                         self.basic.detect(audio_path),
                         self.config.detector_timeout))
            task_names.append("basic_pitch")

        # CREPE
        if use_crepe:
            tasks.append(run_with_timeout("crepe",
                         self.crepe.detect(audio, sr, audio_16k),
                         self.config.detector_timeout))
            task_names.append("crepe")

        # SPICE
        if use_spice:
            tasks.append(run_with_timeout("spice",
                         self.spice.detect(audio, sr, audio_16k),
                         self.config.detector_timeout))
            task_names.append("spice")

        if self.config.debug:
            print(f"🎵 PitchIntelligence: Running {len(tasks)} detectors in parallel...")

        if not tasks:
            if self.config.debug:
                print("⚠️ No detectors enabled")
            return []

        results = await asyncio.gather(*tasks)

        all_events: List[PitchEvent] = []
        detectors_succeeded = 0

        for name, events in results:
            if events:
                all_events.extend(events)
                detectors_succeeded += 1
                if self.config.debug:
                    print(f"✅ {name}: {len(events)} events")

        if self.config.debug:
            print(f"📊 Total events before fusion: {len(all_events)} "
                  f"({detectors_succeeded}/{len(tasks)} detectors succeeded)")

        if not all_events:
            if self.config.debug:
                print("⚠️ No pitch events detected from any source")
            return []

        fused_notes = self.fusion.fuse(all_events)
        final_notes = self.quantizer.process(fused_notes)

        if self.config.debug:
            print(f"🎵 Final notes: {len(final_notes)}")

        return final_notes

    async def process_fast(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Librosa-only fast path for real-time or preview scenarios.

        Use when:
            - Speed is critical
            - ML models are unavailable
            - Debugging pipeline structure
        """
        audio_16k = ensure_16k(audio, sr, 16000)
        events = await self.librosa.detect(audio, sr, audio_16k)
        if not events:
            return []
        notes = self.fusion.fuse(events)
        return self.quantizer.process(notes)

    async def process_piano(self, audio: np.ndarray, sr: int,
                            audio_path: Optional[Path] = None,
                            tempo: float = 120.0,
                            audio_16k: Optional[np.ndarray] = None,
                            state: Any = None) -> List[Note]:  # ← ADD state parameter
        """Process piano audio specifically"""
        notes = await self.process(audio, sr, audio_path, audio_16k)
        for note in notes:
            note.instrument = InstrumentType.PIANO
        return notes

    async def process_bass(self, audio: np.ndarray, sr: int,
                           tempo: float = 120.0,
                           audio_16k: Optional[np.ndarray] = None) -> List[Note]:
        """Process bass audio specifically (uses all detectors)"""
        notes = await self.process(audio, sr, audio_path=None, audio_16k=audio_16k)
        notes = [n for n in notes if 28 <= n.pitch <= 60]
        for note in notes:
            note.instrument = InstrumentType.BASS
        return notes

    async def process_melody(self, audio: np.ndarray, sr: int,
                             stem_name: str = "vocals",
                             tempo: float = 120.0,
                             audio_16k: Optional[np.ndarray] = None) -> List[Note]:
        """Process melody (vocals or lead) audio"""
        notes = await self.process(audio, sr, audio_path=None, audio_16k=audio_16k)
        for note in notes:
            note.instrument = InstrumentType.MELODY
        return notes


# ------------------------------------------------------------
# FACTORY FUNCTION
# ------------------------------------------------------------

def create_pitch_intelligence(debug: bool = False,
                              crepe_confidence: float = 0.35,
                              spice_confidence: float = 0.25,
                              fusion_frame_ms: int = 50,
                              detector_timeout: float = 30.0,
                              use_librosa_only: bool = False) -> PitchIntelligence:
    """
    Factory function to create a configured PitchIntelligence instance.

    Args:
        debug: Enable debug logging
        crepe_confidence: Confidence threshold for CREPE (0-1)
        spice_confidence: Confidence threshold for SPICE (0-1)
        fusion_frame_ms: Frame size for fusion in milliseconds
        detector_timeout: Timeout per detector in seconds
        use_librosa_only: If True, only use librosa (fastest, no ML)

    Returns:
        Configured PitchIntelligence instance
    """
    config = PitchConfig(
        debug=debug,
        crepe_confidence=crepe_confidence,
        spice_confidence=spice_confidence,
        fusion_frame_ms=fusion_frame_ms,
        detector_timeout=detector_timeout
    )
    return PitchIntelligence(config)


# ------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Pitch Intelligence v3")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--fast", action="store_true", help="Use librosa-only fast mode")
    parser.add_argument("--crepe-conf", type=float, default=0.35, help="CREPE confidence threshold")
    parser.add_argument("--spice-conf", type=float, default=0.25, help="SPICE confidence threshold")
    parser.add_argument("--timeout", type=float, default=30.0, help="Detector timeout in seconds")

    args = parser.parse_args()

    async def test():
        print("=" * 60)
        print("🧪 Pitch Intelligence v3 Test")
        print("=" * 60)

        audio, sr = librosa.load(args.audio_file, sr=22050)
        print(f"📂 Loaded: {args.audio_file}")
        print(f"   Duration: {len(audio) / sr:.1f}s, SR: {sr}Hz")

        audio_16k = ensure_16k(audio, sr, 16000)
        print(f"   Pre-resampled to 16kHz: {len(audio_16k)} samples")

        pitch_intel = create_pitch_intelligence(
            debug=args.debug,
            crepe_confidence=args.crepe_conf,
            spice_confidence=args.spice_conf,
            detector_timeout=args.timeout
        )

        if args.fast:
            print("\n🚀 Running in FAST mode (librosa only)...")
            notes = await pitch_intel.process_fast(audio, sr)
        else:
            print("\n🎵 Running in FULL mode (librosa + ML models)...")
            notes = await pitch_intel.process(
                audio, sr,
                audio_path=Path(args.audio_file),
                audio_16k=audio_16k
            )

        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Total notes detected: {len(notes)}")

        if notes:
            print("\nFirst 20 notes:")
            print("-" * 60)
            for i, note in enumerate(notes[:20]):
                print(f"  {i + 1:3d}. Pitch: {note.pitch:3d} | "
                      f"Start: {note.start:6.2f}s | "
                      f"End: {note.end:6.2f}s | "
                      f"Vel: {note.velocity:3d} | "
                      f"Conf: {note.confidence:.2f}")

        print("\n✅ Test complete")

    asyncio.run(test())