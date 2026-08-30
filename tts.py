"""TTS engines behind one interface, selected with TTS_ENGINE=kitten|kokoro.

- kitten: KittenTTS Nano (~56MB, 15M params). Lightest option.
- kokoro: Kokoro-82M via kokoro-onnx. Much more natural voices; the int8
  variant (~114MB model) still fits Render Free's 512MB RAM.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("podcast-mcp.tts")

SAMPLE_RATE = 24000

# Pause lengths in seconds
CHUNK_PAUSE = 0.18   # between chunks of one speaker turn
TURN_PAUSE = 0.45    # between speaker turns


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
        path = self.model_dir / filename
        if path.exists():
            return path
        self.model_dir.mkdir(parents=True, exist_ok=True)
        url = f"{_KOKORO_RELEASE}/{filename}"
        logger.info("Downloading %s ...", url)
        started = time.time()
        temp_path = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temp_path)
        temp_path.replace(path)
        logger.info(
            "Downloaded %s (%.0f MB) in %.0fs",
            filename, path.stat().st_size / 1e6, time.time() - started,
        )
        return path

    def load(self):
        if self._model is None:
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


_BACKENDS = {"kitten": KittenBackend, "kokoro": KokoroBackend}


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

        segments = []
        with self._lock:
            for turn_index, (voice, chunks) in enumerate(turns):
                if turn_index > 0:
                    segments.append(turn_gap)
                for chunk_index, chunk in enumerate(chunks):
                    if chunk_index > 0:
                        segments.append(chunk_gap)
                    segments.append(self.backend.generate_chunk(chunk, voice, speed))
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
