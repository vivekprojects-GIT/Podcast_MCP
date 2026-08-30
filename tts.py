"""TTS engines behind one interface, selected with TTS_ENGINE=piper|kitten|kokoro.

- piper: Piper TTS (rhasspy voices, ~63MB each). Clear voices with a small
  inference footprint — the only engine that fits Render Free's 512MB RAM
  under real load (measured: ~230MB steady, ~370MB peak).
- kitten: KittenTTS Nano. Small download but onnxruntime peaks at ~550MB
  during generation — fine locally, too big for Render Free.
- kokoro: Kokoro-82M via kokoro-onnx. Most natural voices; needs a paid
  instance.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
import urllib.request
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("podcast-mcp.tts")

SAMPLE_RATE = 24000

# Pause lengths in seconds
CHUNK_PAUSE = 0.18   # between chunks of one speaker turn
TURN_PAUSE = 0.45    # between speaker turns


# Pause between chunk generations so the event loop can answer health checks
# on tiny (0.1 CPU) instances — without it, Render marks the service unhealthy
# and restarts it mid-generation.
_YIELD_SLEEP = float(os.environ.get("TTS_YIELD_SLEEP", "0.15"))


def _deprioritize_current_thread() -> None:
    """Lower this thread's scheduling priority (Linux) so the web server wins."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except (AttributeError, OSError):
        pass  # not Linux, or not permitted — best effort only


_ORT_PATCHED = False


def _low_memory_onnxruntime() -> None:
    """Disable onnxruntime's CPU memory arena and cap its threads.

    The arena retains peak activation memory and over-allocates in
    power-of-two steps, which is what pushed every engine past Render Free's
    512MB limit. Set LOW_MEMORY_MODE=0 on bigger instances for more speed.
    """
    global _ORT_PATCHED
    if _ORT_PATCHED or os.environ.get("LOW_MEMORY_MODE", "1") != "1":
        return
    import onnxruntime as ort

    original_session = ort.InferenceSession

    class LowMemorySession(original_session):
        def __init__(self, *args, **kwargs):
            options = kwargs.get("sess_options") or ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            kwargs["sess_options"] = options
            super().__init__(*args, **kwargs)

    ort.InferenceSession = LowMemorySession
    _ORT_PATCHED = True
    logger.info("onnxruntime patched for low memory (no arena, single thread)")


def _download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)
    started = time.time()
    temp_path = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temp_path)
    temp_path.replace(path)
    logger.info(
        "Downloaded %s (%.0f MB) in %.0fs",
        path.name, path.stat().st_size / 1e6, time.time() - started,
    )
    return path


def _resample(samples, source_rate: int):
    """Linear resample to SAMPLE_RATE (engines are expected to match already)."""
    if source_rate == SAMPLE_RATE:
        return samples
    import numpy as np

    target_length = int(len(samples) * SAMPLE_RATE / source_rate)
    positions_old = np.linspace(0.0, 1.0, len(samples))
    positions_new = np.linspace(0.0, 1.0, target_length)
    return np.interp(positions_new, positions_old, samples).astype(np.float32)


class KittenBackend:
    name = "kitten"
    voices = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
    default_host_voice = "Jasper"
    default_guest_voice = "Bella"
    strict_voices = True

    def __init__(self) -> None:
        # Full-precision nano (~56MB). The int8 variant is smaller but has
        # known quality issues (muffled/slurred output) per the KittenTTS repo.
        self.model_id = os.environ.get("KITTEN_MODEL", "KittenML/kitten-tts-nano-0.8")
        self._model = None

    def load(self):
        if self._model is None:
            _low_memory_onnxruntime()
            from kittentts import KittenTTS  # heavy import, keep lazy

            started = time.time()
            logger.info("Loading KittenTTS model '%s'...", self.model_id)
            self._model = KittenTTS(self.model_id)
            logger.info("KittenTTS loaded in %.1fs", time.time() - started)
        return self._model

    def generate_chunk(self, text: str, voice: str, speed: float):
        import numpy as np

        audio = self.load().generate(text, voice=voice, speed=speed)
        return np.asarray(audio, dtype=np.float32)


_KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
_KOKORO_MODELS = {
    "int8": "kokoro-v1.0.int8.onnx",  # 114 MB — fits Render Free RAM
    "fp16": "kokoro-v1.0.fp16.onnx",  # 164 MB — near-lossless, needs more RAM
    "fp32": "kokoro-v1.0.onnx",       # 326 MB — paid instances only
}
_KOKORO_VOICES_BIN = "voices-v1.0.bin"


class KokoroBackend:
    name = "kokoro"
    # English voices from the v1.0 54-voice pack (af/am = American, bf/bm = British)
    voices = [
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_alloy",
        "af_aoede", "af_jessica", "af_kore", "af_nova", "af_river",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ]
    default_host_voice = "am_michael"
    default_guest_voice = "af_heart"
    strict_voices = False  # other languages exist in the pack; Kokoro validates at runtime

    def __init__(self) -> None:
        self.variant = os.environ.get("KOKORO_VARIANT", "int8").lower()
        if self.variant not in _KOKORO_MODELS:
            raise ValueError(
                f"KOKORO_VARIANT must be one of {sorted(_KOKORO_MODELS)}, got '{self.variant}'"
            )
        self.model_dir = Path(os.environ.get("KOKORO_MODEL_DIR", "models/kokoro"))
        self.model_id = f"kokoro-82m-{self.variant}"
        self._model = None

    def _fetch(self, filename: str) -> Path:
        return _download(f"{_KOKORO_RELEASE}/{filename}", self.model_dir / filename)

    def load(self):
        if self._model is None:
            _low_memory_onnxruntime()
            from kokoro_onnx import Kokoro  # heavy import, keep lazy

            model_path = self._fetch(_KOKORO_MODELS[self.variant])
            voices_path = self._fetch(_KOKORO_VOICES_BIN)
            started = time.time()
            logger.info("Loading Kokoro model '%s'...", self.model_id)
            self._model = Kokoro(str(model_path), str(voices_path))
            logger.info("Kokoro loaded in %.1fs", time.time() - started)
        return self._model

    def generate_chunk(self, text: str, voice: str, speed: float):
        import numpy as np

        lang = "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"
        samples, source_rate = self.load().create(text, voice=voice, speed=speed, lang=lang)
        return _resample(np.asarray(samples, dtype=np.float32), source_rate)


_PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


class PiperBackend:
    name = "piper"
    # Curated en voices from rhasspy/piper-voices; any "<locale>-<name>-<quality>"
    # voice from that repo also works (downloaded on demand).
    voices = [
        "en_US-hfc_male-medium", "en_US-hfc_female-medium",
        "en_US-ryan-medium", "en_US-amy-medium", "en_US-lessac-medium",
        "en_US-joe-medium", "en_US-kristin-medium", "en_US-kusal-medium",
        "en_GB-alan-medium", "en_GB-cori-medium", "en_GB-jenny_dioco-medium",
        "en_GB-northern_english_male-medium",
    ]
    default_host_voice = "en_US-hfc_male-medium"
    default_guest_voice = "en_US-hfc_female-medium"
    strict_voices = False
    _MAX_LOADED = 2  # host + guest; keeps two ~130MB sessions max in RAM

    def __init__(self) -> None:
        self.model_dir = Path(os.environ.get("PIPER_MODEL_DIR", "models/piper"))
        self.model_id = "piper-1.0-medium"
        self._loaded: OrderedDict[str, object] = OrderedDict()

    def _fetch_voice(self, voice: str) -> Path:
        parts = voice.split("-")
        if len(parts) != 3 or "_" not in parts[0]:
            raise ValueError(
                f"Piper voice must look like 'en_US-ryan-medium', got '{voice}'"
            )
        locale, name, quality = parts
        base = f"{_PIPER_VOICES_BASE}/{locale.split('_')[0]}/{locale}/{name}/{quality}"
        model_path = _download(f"{base}/{voice}.onnx", self.model_dir / f"{voice}.onnx")
        _download(f"{base}/{voice}.onnx.json", self.model_dir / f"{voice}.onnx.json")
        return model_path

    def _get_voice(self, voice: str):
        if voice in self._loaded:
            self._loaded.move_to_end(voice)
            return self._loaded[voice]
        _low_memory_onnxruntime()
        from piper import PiperVoice  # heavy import, keep lazy

        model_path = self._fetch_voice(voice)
        started = time.time()
        loaded = PiperVoice.load(str(model_path))
        logger.info("Piper voice '%s' loaded in %.1fs", voice, time.time() - started)
        self._loaded[voice] = loaded
        while len(self._loaded) > self._MAX_LOADED:
            evicted, _ = self._loaded.popitem(last=False)
            logger.info("Evicted Piper voice '%s' to save memory", evicted)
            gc.collect()
        return loaded

    def load(self):
        self._get_voice(self.default_host_voice)
        self._get_voice(self.default_guest_voice)

    def generate_chunk(self, text: str, voice: str, speed: float):
        import numpy as np
        from piper import SynthesisConfig

        model = self._get_voice(voice)
        config = SynthesisConfig(length_scale=1.0 / max(speed, 0.1))
        pieces = []
        sample_rate = SAMPLE_RATE
        for chunk in model.synthesize(text, syn_config=config):
            sample_rate = chunk.sample_rate
            pieces.append(
                np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        if not pieces:
            raise ValueError(f"Piper produced no audio for: {text[:60]!r}")
        return _resample(np.concatenate(pieces), sample_rate)


_BACKENDS = {"kitten": KittenBackend, "kokoro": KokoroBackend, "piper": PiperBackend}


class TTSEngine:
    """Serialized access to a lazily-loaded TTS backend (Render Free has 1 CPU)."""

    def __init__(self) -> None:
        backend_name = os.environ.get("TTS_ENGINE", "kitten").lower()
        if backend_name not in _BACKENDS:
            raise ValueError(
                f"TTS_ENGINE must be one of {sorted(_BACKENDS)}, got '{backend_name}'"
            )
        self.backend = _BACKENDS[backend_name]()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    @property
    def voices(self) -> list[str]:
        return self.backend.voices

    @property
    def default_host_voice(self) -> str:
        return self.backend.default_host_voice

    @property
    def default_guest_voice(self) -> str:
        return self.backend.default_guest_voice

    def resolve_voice(self, name: str) -> str:
        name = name.strip()
        for voice in self.backend.voices:
            if voice.lower() == name.lower():
                return voice
        if self.backend.strict_voices:
            raise ValueError(
                f"Unknown voice '{name}' for engine '{self.name}'. "
                f"Available voices: {', '.join(self.backend.voices)}"
            )
        return name  # let the backend validate uncommon voices itself

    def preload(self) -> None:
        try:
            _deprioritize_current_thread()
            with self._lock:
                self.backend.load()
            logger.info("TTS engine '%s' (%s) preloaded", self.name, self.model_id)
        except Exception:
            logger.exception("Model preload failed; will retry on first request")

    def synthesize_turns(
        self, turns: list[tuple[str, list[str]]], speed: float = 1.0
    ):
        """Synthesize [(voice, [chunk, ...]), ...] into one float32 waveform."""
        import numpy as np

        chunk_gap = np.zeros(int(CHUNK_PAUSE * SAMPLE_RATE), dtype=np.float32)
        turn_gap = np.zeros(int(TURN_PAUSE * SAMPLE_RATE), dtype=np.float32)

        _deprioritize_current_thread()
        segments = []
        with self._lock:
            for turn_index, (voice, chunks) in enumerate(turns):
                if turn_index > 0:
                    segments.append(turn_gap)
                for chunk_index, chunk in enumerate(chunks):
                    if chunk_index > 0:
                        segments.append(chunk_gap)
                    segments.append(self.backend.generate_chunk(chunk, voice, speed))
                    if _YIELD_SLEEP > 0:
                        time.sleep(_YIELD_SLEEP)  # let the event loop answer health checks
                gc.collect()  # keep RSS flat on memory-tight instances
        if not segments:
            raise ValueError("Nothing to synthesize")
        waveform = np.concatenate(segments)
        return np.clip(waveform, -1.0, 1.0)


def encode_mp3(waveform, path: Path, bitrate_kbps: int = 128) -> None:
    import lameenc
    import numpy as np

    pcm = (waveform * 32767.0).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    data = encoder.encode(pcm.tobytes())
    data += encoder.flush()
    path.write_bytes(bytes(data))


def encode_wav(waveform, path: Path) -> None:
    import soundfile as sf

    sf.write(str(path), waveform, SAMPLE_RATE)
