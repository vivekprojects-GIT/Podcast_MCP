"""Podcast MCP server.

Turns a host/guest podcast script (written by YOUR app's LLM) into a single
MP3 using KittenTTS Nano, then returns a public audio URL.

Transport: MCP streamable HTTP at /mcp (stateless), plus plain HTTP routes:
    GET /health            -> {"status": "ok"}
    GET /audio/<filename>  -> serves generated audio files
"""

from __future__ import annotations

import functools
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from script_parser import parse_script, split_into_chunks
from tts import SAMPLE_RATE, TTSEngine, encode_mp3, encode_wav
from video import build_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("podcast-mcp")

PORT = int(os.environ.get("PORT", "8000"))
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "audio_output")).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_TTL_HOURS = float(os.environ.get("AUDIO_TTL_HOURS", "24"))
MAX_SCRIPT_CHARS = int(os.environ.get("MAX_SCRIPT_CHARS", "20000"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "300"))
MAX_VIDEO_SECTIONS = int(os.environ.get("MAX_VIDEO_SECTIONS", "20"))

engine = TTSEngine()

# Empty string means "use the engine's default voice"
DEFAULT_HOST_VOICE = os.environ.get("DEFAULT_HOST_VOICE", "")
DEFAULT_GUEST_VOICE = os.environ.get("DEFAULT_GUEST_VOICE", "")

mcp = FastMCP(
    "podcast-mcp",
    instructions=(
        "Generate podcast audio and narrated slide videos from content your own "
        "LLM writes. Call generate_podcast_from_script with dialogue lines like "
        "'HOST: ...' and 'GUEST: ...' for an MP3; generate_video_from_sections "
        "with structured sections (title, narration, key_points, optional "
        "bar_chart visual) for an MP4; text_to_speech for single-voice audio. "
        "All return a public URL."
    ),
    host="0.0.0.0",
    port=PORT,
    stateless_http=True,
    json_response=True,
)


_STARTED_AT = time.time()


def _rss_mb() -> float | None:
    """Resident memory in MB (Linux only) — for watching the 512MB free-tier limit."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def _public_base_url() -> str:
    return (
        os.environ.get("PUBLIC_BASE_URL")
        or os.environ.get("RENDER_EXTERNAL_URL")  # set automatically by Render
        or f"http://localhost:{PORT}"
    ).rstrip("/")


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "audio"


def _new_audio_path(title: str, extension: str) -> Path:
    return AUDIO_DIR / f"{_slugify(title)}-{uuid.uuid4().hex[:8]}.{extension}"


def _cleanup_old_files() -> None:
    cutoff = time.time() - AUDIO_TTL_HOURS * 3600
    for file in AUDIO_DIR.glob("*"):
        try:
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
        except OSError:
            pass


def _assign_voices(speakers: list[str], host_voice: str, guest_voice: str) -> dict[str, str]:
    """HOST/GUEST get their configured voices; extra named speakers cycle the rest."""
    assignments: dict[str, str] = {"HOST": host_voice, "GUEST": guest_voice}
    pool = [host_voice, guest_voice] + [
        v for v in engine.voices if v not in (host_voice, guest_voice)
    ]
    next_index = 0
    for speaker in speakers:
        if speaker not in assignments:
            assignments[speaker] = pool[next_index % len(pool)]
            next_index += 1
    return assignments


@mcp.tool()
async def generate_podcast_from_script(
    script: str,
    title: str = "",
    host_voice: str = "",
    guest_voice: str = "",
    speed: float = 1.0,
) -> dict[str, Any]:
    """Convert a host/guest dialogue script into a single podcast MP3.

    The script should contain lines labeled with speakers, e.g.:
        HOST: Welcome back to the show. Today we're covering the Q2 numbers.
        GUEST: Thanks! The headline is revenue up 18 percent.
    Any speaker labels work (SARAH:, MIKE:, ...); the first two speakers get
    host_voice and guest_voice. Leave voices empty for the engine defaults;
    call list_voices to see what's available. Returns a public audio_url.
    """
    # Off the event loop: synthesis takes minutes on small instances, and the
    # MCP SDK runs sync tools inline, which would block health checks.
    return await anyio.to_thread.run_sync(
        functools.partial(
            _generate_podcast_sync, script, title, host_voice, guest_voice, speed
        )
    )


def _generate_podcast_sync(
    script: str, title: str, host_voice: str, guest_voice: str, speed: float
) -> dict[str, Any]:
    try:
        if len(script) > MAX_SCRIPT_CHARS:
            return {
                "success": False,
                "error": f"Script too long ({len(script)} chars, max {MAX_SCRIPT_CHARS}).",
            }
        host_voice = engine.resolve_voice(
            host_voice or DEFAULT_HOST_VOICE or engine.default_host_voice
        )
        guest_voice = engine.resolve_voice(
            guest_voice or DEFAULT_GUEST_VOICE or engine.default_guest_voice
        )

        turns = parse_script(script)
        if not turns:
            return {"success": False, "error": "No dialogue lines found in script."}

        speakers_in_order = list(dict.fromkeys(turn.speaker for turn in turns))
        voice_map = _assign_voices(speakers_in_order, host_voice, guest_voice)

        synth_input = [
            (voice_map[turn.speaker], split_into_chunks(turn.text, MAX_CHUNK_CHARS))
            for turn in turns
        ]

        started = time.time()
        waveform = engine.synthesize_turns(synth_input, speed=speed)
        path = _new_audio_path(title or "podcast", "mp3")
        encode_mp3(waveform, path)
        _cleanup_old_files()

        duration = round(len(waveform) / SAMPLE_RATE, 1)
        logger.info(
            "Generated %s: %.1fs audio from %d turns in %.1fs",
            path.name, duration, len(turns), time.time() - started,
        )
        return {
            "success": True,
            "type": "podcast",
            "title": title or "Podcast",
            "audio_url": f"{_public_base_url()}/audio/{path.name}",
            "duration_seconds": duration,
            "turns": len(turns),
            "voices": {speaker: voice_map[speaker] for speaker in speakers_in_order},
        }
    except Exception as exc:  # tool errors go back to the caller as data
        logger.exception("generate_podcast_from_script failed")
        return {"success": False, "error": str(exc)}


@mcp.tool()
async def text_to_speech(
    text: str,
    voice: str = "",
    speed: float = 1.0,
    format: str = "mp3",
) -> dict[str, Any]:
    """Convert plain text to speech with a single voice. Returns a public audio_url.

    Leave voice empty for the engine default; call list_voices for options.
    format: "mp3" or "wav".
    """
    return await anyio.to_thread.run_sync(
        functools.partial(_text_to_speech_sync, text, voice, speed, format)
    )


def _text_to_speech_sync(text: str, voice: str, speed: float, format: str) -> dict[str, Any]:
    try:
        if len(text) > MAX_SCRIPT_CHARS:
            return {
                "success": False,
                "error": f"Text too long ({len(text)} chars, max {MAX_SCRIPT_CHARS}).",
            }
        if format not in ("mp3", "wav"):
            return {"success": False, "error": "format must be 'mp3' or 'wav'."}
        voice = engine.resolve_voice(voice or DEFAULT_HOST_VOICE or engine.default_host_voice)

        chunks = split_into_chunks(text, MAX_CHUNK_CHARS)
        if not chunks:
            return {"success": False, "error": "No text to synthesize."}

        waveform = engine.synthesize_turns([(voice, chunks)], speed=speed)
        path = _new_audio_path(text[:40], format)
        (encode_mp3 if format == "mp3" else encode_wav)(waveform, path)
        _cleanup_old_files()

        return {
            "success": True,
            "type": "speech",
            "audio_url": f"{_public_base_url()}/audio/{path.name}",
            "duration_seconds": round(len(waveform) / SAMPLE_RATE, 1),
            "voice": voice,
        }
    except Exception as exc:
        logger.exception("text_to_speech failed")
        return {"success": False, "error": str(exc)}


@mcp.tool()
async def generate_video_from_sections(
    sections: list[dict[str, Any]],
    title: str = "",
    voice: str = "",
    speed: float = 1.0,
) -> dict[str, Any]:
    """Render a narrated slide video (MP4) from structured report sections.

    Each section: {"title": str, "narration": str, "key_points": [str],
    "visual": {"type": "bar_chart" | "bullet_summary", "data": {label: number}}}.
    Every section becomes one slide (title + bullets + optional bar chart)
    shown for the length of its narration, spoken by a single voice (empty =
    engine default). Pass the same sections you use for the podcast script to
    keep both outputs consistent. Returns a public video_url.
    """
    return await anyio.to_thread.run_sync(
        functools.partial(_generate_video_sync, sections, title, voice, speed)
    )


def _generate_video_sync(
    sections: list[dict[str, Any]], title: str, voice: str, speed: float
) -> dict[str, Any]:
    try:
        if not isinstance(sections, list) or not sections:
            return {"success": False, "error": "sections must be a non-empty list."}
        if len(sections) > MAX_VIDEO_SECTIONS:
            return {
                "success": False,
                "error": f"Too many sections ({len(sections)}, max {MAX_VIDEO_SECTIONS}).",
            }
        total_chars = sum(len(str(s.get("narration", ""))) for s in sections)
        if total_chars > MAX_SCRIPT_CHARS:
            return {
                "success": False,
                "error": f"Narration too long ({total_chars} chars, max {MAX_SCRIPT_CHARS}).",
            }
        voice = engine.resolve_voice(voice or DEFAULT_HOST_VOICE or engine.default_host_voice)

        started = time.time()
        path = _new_audio_path(title or "video", "mp4")
        duration = build_video(
            engine, sections, title, voice, speed, path, MAX_CHUNK_CHARS
        )
        _cleanup_old_files()

        logger.info(
            "Generated %s: %.1fs video from %d sections in %.1fs",
            path.name, duration, len(sections), time.time() - started,
        )
        return {
            "success": True,
            "type": "video",
            "title": title or "Video",
            "video_url": f"{_public_base_url()}/audio/{path.name}",
            "duration_seconds": round(duration, 1),
            "sections": len(sections),
            "voice": voice,
        }
    except Exception as exc:
        logger.exception("generate_video_from_sections failed")
        return {"success": False, "error": str(exc)}


@mcp.tool()
def list_voices() -> dict[str, Any]:
    """List the active TTS engine, its available voices, and the current defaults."""
    return {
        "engine": engine.name,
        "model": engine.model_id,
        "voices": engine.voices,
        "default_host_voice": DEFAULT_HOST_VOICE or engine.default_host_voice,
        "default_guest_voice": DEFAULT_GUEST_VOICE or engine.default_guest_voice,
        "sample_rate": SAMPLE_RATE,
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "engine": engine.name,
            "model": engine.model_id,
            "uptime_seconds": round(time.time() - _STARTED_AT),
            "rss_mb": _rss_mb(),
        }
    )


@mcp.custom_route("/", methods=["GET"])
async def index(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "service": "podcast-mcp",
            "mcp_endpoint": "/mcp",
            "tools": [
                "generate_podcast_from_script",
                "generate_video_from_sections",
                "text_to_speech",
                "list_voices",
            ],
        }
    )


@mcp.custom_route("/audio/{filename}", methods=["GET"])
async def serve_audio(request: Request) -> FileResponse | JSONResponse:
    filename = Path(request.path_params["filename"]).name  # strip any path parts
    path = (AUDIO_DIR / filename).resolve()
    if not path.is_file() or path.parent != AUDIO_DIR:
        return JSONResponse({"error": "not found"}, status_code=404)
    media_type = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
    }.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


if __name__ == "__main__":
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        threading.Thread(target=engine.preload, daemon=True).start()
    logger.info("Starting podcast-mcp on port %d (MCP endpoint: /mcp)", PORT)
    mcp.run(transport="streamable-http")
