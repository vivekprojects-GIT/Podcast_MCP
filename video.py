"""Render narrated slide videos (MP4) from structured report sections.

Each section becomes one slide (title, bullet points, optional bar chart —
drawn with PIL, no matplotlib) shown for the length of its narration, which is
synthesized with the active TTS engine. ffmpeg muxes stills + narration into
an MP4; still-image encoding keeps CPU and RAM small enough for Render Free.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tts import SAMPLE_RATE, TTSEngine
from script_parser import split_into_chunks

logger = logging.getLogger("podcast-mcp.video")

WIDTH, HEIGHT = 1280, 720
BG = (15, 23, 42)         # slate-900
FG = (241, 245, 249)      # slate-100
MUTED = (148, 163, 184)   # slate-400
ACCENT = (56, 189, 248)   # sky-400
BAR_ALT = (100, 116, 139)  # slate-500

SECTION_PAUSE = 0.7  # silence appended after each section's narration

# Branding watermark shown bottom-right on every slide; point LOGO_PATH at a
# transparent PNG (relative paths resolve against this file's directory), or
# set LOGO_PATH="" to disable.
LOGO_PATH = os.environ.get("LOGO_PATH", "assets/logo.png")
LOGO_HEIGHT = 34

_logo_cache: list = []  # [Image | None], lazily filled

_FONTS_BOLD = [
    "DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]
_FONTS_REGULAR = [
    "DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in _FONTS_BOLD if bold else _FONTS_REGULAR:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _get_logo():
    if not _logo_cache:
        logo = None
        if LOGO_PATH:
            path = Path(LOGO_PATH)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if path.is_file():
                try:
                    from PIL import Image

                    logo = Image.open(path).convert("RGBA")
                    scale = LOGO_HEIGHT / logo.height
                    logo = logo.resize(
                        (max(1, int(logo.width * scale)), LOGO_HEIGHT),
                        Image.LANCZOS,
                    )
                except Exception:
                    logger.exception("Could not load logo from %s", path)
                    logo = None
            else:
                logger.warning("LOGO_PATH %s not found; slides get no logo", path)
        _logo_cache.append(logo)
    return _logo_cache[0]


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_bar_chart(draw, data: dict, x: int, y: int, width: int, height: int) -> None:
    items = []
    for label, value in list(data.items())[:6]:
        try:
            items.append((str(label), float(value)))
        except (TypeError, ValueError):
            continue
    if not items:
        return
    max_abs = max(abs(value) for _, value in items) or 1.0
    label_font = _font(26)
    value_font = _font(26, bold=True)
    label_width = 300
    bar_height = min(48, max(24, height // len(items) - 22))
    for index, (label, value) in enumerate(items):
        bar_y = y + index * (bar_height + 22)
        if bar_y + bar_height > y + height:
            break
        draw.text((x, bar_y + bar_height // 2), label[:24], font=label_font,
                  fill=MUTED, anchor="lm")
        bar_length = int((width - label_width - 130) * abs(value) / max_abs)
        color = ACCENT if index == 0 else BAR_ALT
        draw.rounded_rectangle(
            [x + label_width, bar_y, x + label_width + max(bar_length, 4), bar_y + bar_height],
            radius=6, fill=color,
        )
        draw.text(
            (x + label_width + max(bar_length, 4) + 14, bar_y + bar_height // 2),
            f"{value:g}", font=value_font, fill=FG, anchor="lm",
        )


def render_slide(section: dict, index: int, total: int, deck_title: str, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, 8, HEIGHT], fill=ACCENT)
    if deck_title:
        draw.text((60, 40), deck_title[:80], font=_font(24), fill=MUTED)
    draw.text((WIDTH - 60, 40), f"{index + 1} / {total}", font=_font(24),
              fill=MUTED, anchor="ra")

    y = 110
    title_font = _font(46, bold=True)
    for line in _wrap(draw, str(section.get("title", "")) or f"Section {index + 1}",
                      title_font, WIDTH - 140)[:2]:
        draw.text((60, y), line, font=title_font, fill=FG)
        y += 62
    y += 24

    bullet_font = _font(30)
    for point in [str(p) for p in section.get("key_points", []) if str(p).strip()][:6]:
        for line_index, line in enumerate(_wrap(draw, point, bullet_font, WIDTH - 220)[:2]):
            if line_index == 0:
                draw.ellipse([64, y + 12, 78, y + 26], fill=ACCENT)
            draw.text((100, y), line, font=bullet_font, fill=FG)
            y += 44
        y += 10

    visual = section.get("visual") or {}
    if visual.get("type") == "bar_chart" and isinstance(visual.get("data"), dict):
        chart_y = max(y + 30, 330)
        _draw_bar_chart(draw, visual["data"], 80, chart_y, WIDTH - 160,
                        HEIGHT - chart_y - 90)

    logo = _get_logo()
    if logo is not None:
        pad_x, pad_y = 14, 9
        chip_w = logo.width + pad_x * 2
        chip_h = logo.height + pad_y * 2
        chip_x = WIDTH - 36 - chip_w
        chip_y = HEIGHT - 28 - chip_h
        draw.rounded_rectangle(
            [chip_x, chip_y, chip_x + chip_w, chip_y + chip_h],
            radius=10, fill=(248, 250, 252),
        )
        image.paste(logo, (chip_x + pad_x, chip_y + pad_y), logo)

    image.save(path, "PNG")


def render_podcast_cover(title: str, path: Path) -> None:
    """Square branded cover art for podcast MP3s (embedded ID3 + thumbnail_url)."""
    from PIL import Image, ImageDraw

    size = 1000
    image = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, 10, size], fill=ACCENT)
    draw.text((80, 110), " ".join("PODCAST"), font=_font(30, bold=True), fill=ACCENT)

    y = 190
    title_font = _font(72, bold=True)
    for line in _wrap(draw, title or "Podcast", title_font, size - 170)[:4]:
        draw.text((80, y), line, font=title_font, fill=FG)
        y += 92

    # sound-wave motif
    heights = [40, 95, 60, 130, 80, 160, 105, 70, 140, 90, 55, 115, 75, 150,
               100, 65, 125, 85, 50, 110, 70, 145, 95, 60]
    baseline = 780
    x = 80
    for index, bar_height in enumerate(heights):
        color = ACCENT if index % 3 == 0 else BAR_ALT
        draw.rounded_rectangle(
            [x, baseline - bar_height // 2, x + 14, baseline + bar_height // 2],
            radius=7, fill=color,
        )
        x += 26

    logo = _get_logo()
    if logo is not None:
        scale = 44 / logo.height
        big_logo = logo.resize((max(1, int(logo.width * scale)), 44))
        pad_x, pad_y = 18, 12
        chip_w = big_logo.width + pad_x * 2
        chip_h = big_logo.height + pad_y * 2
        chip_x = size - 60 - chip_w
        chip_y = size - 60 - chip_h
        draw.rounded_rectangle(
            [chip_x, chip_y, chip_x + chip_w, chip_y + chip_h],
            radius=12, fill=(248, 250, 252),
        )
        image.paste(big_logo, (chip_x + pad_x, chip_y + pad_y), big_logo)

    image.save(path, "JPEG", quality=88)


def embed_cover(mp3_path: Path, cover_path: Path, title: str) -> None:
    """Embed cover art + title into the MP3's ID3 tags."""
    from mutagen.id3 import APIC, ID3, TALB, TIT2

    tags = ID3()
    tags.add(TIT2(encoding=3, text=title or "Podcast"))
    tags.add(TALB(encoding=3, text="Generated by Podcast MCP"))
    tags.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover",
                  data=cover_path.read_bytes()))
    tags.save(mp3_path)


def _section_narration(section: dict, index: int) -> str:
    narration = str(section.get("narration", "")).strip()
    if narration:
        return narration
    parts = [str(section.get("title", "")).strip()]
    parts += [str(p).strip() for p in section.get("key_points", [])]
    text = ". ".join(p for p in parts if p)
    if not text:
        raise ValueError(f"Section {index + 1} has no narration, title, or key_points")
    return text


def build_video(
    engine: TTSEngine,
    sections: list[dict],
    deck_title: str,
    voice: str,
    speed: float,
    output_path: Path,
    max_chunk_chars: int = 300,
) -> float:
    """Render sections into an MP4 at output_path; returns duration in seconds."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is not installed. Video generation needs ffmpeg on PATH "
            "(included in the Docker image)."
        )

    with tempfile.TemporaryDirectory(prefix="video_") as workdir_str:
        workdir = Path(workdir_str)
        pause = np.zeros(int(SECTION_PAUSE * SAMPLE_RATE), dtype=np.float32)
        durations: list[float] = []

        # NARRATION IS STREAMED TO DISK, NEVER ASSEMBLED IN MEMORY.
        #
        # This held every section's audio in a list and then called
        # np.concatenate over it, which is a second full copy of the whole
        # narration alive at the same moment as the first. On the free 512MB
        # instance, with piper already resident, that peak plus the ffmpeg
        # subprocess spawned immediately afterwards was enough to have the
        # container killed - and a killed container is a 502 with no error
        # anywhere, which is exactly how this failed: reproducibly, on a
        # single section of ten words, after surviving just long enough to
        # synthesise the audio.
        #
        # soundfile can append, so each section is written and released as
        # it is made. Peak audio memory is now ONE SECTION rather than the
        # whole deck, and it no longer grows with the length of the video.
        import soundfile as sf

        narration_path = workdir / "narration.wav"
        with sf.SoundFile(str(narration_path), "w", SAMPLE_RATE, 1,
                          "PCM_16") as out:
            for index, section in enumerate(sections):
                slide_path = workdir / f"slide_{index:03d}.png"
                render_slide(section, index, len(sections), deck_title, slide_path)

                narration = _section_narration(section, index)
                chunks = split_into_chunks(narration, max_chunk_chars)
                waveform = engine.synthesize_turns([(voice, chunks)], speed=speed)
                out.write(waveform)
                out.write(pause)
                durations.append((len(waveform) + len(pause)) / SAMPLE_RATE)
                logger.info("Section %d/%d: %.1fs narration", index + 1,
                            len(sections), durations[-1])
                del waveform
                gc.collect()

        # RELEASE THE VOICE BEFORE FORKING.
        #
        # subprocess.run forks this process, and what it forks is one still
        # holding a loaded piper model - the single largest allocation in the
        # service at roughly 230MB of a 512MB budget. Dropping the LRU here
        # hands that back before ffmpeg needs its own, and the next request
        # reloads the voice lazily in about a second and a half. Paying that
        # occasionally is plainly better than the render failing.
        loaded = getattr(getattr(engine, "backend", None), "_loaded", None)
        if hasattr(loaded, "clear"):
            try:
                loaded.clear()
                logger.info("Released TTS voices before encoding")
            except Exception:      # never fail a render over housekeeping
                pass
        gc.collect()

        concat_path = workdir / "slides.txt"
        lines = []
        for index, duration in enumerate(durations):
            lines.append(f"file 'slide_{index:03d}.png'")
            lines.append(f"duration {duration:.3f}")
        lines.append(f"file 'slide_{len(durations) - 1:03d}.png'")  # concat demuxer quirk
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        command = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-i", str(narration_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
            "-r", "4", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
        if os.name == "posix" and shutil.which("nice"):
            command = ["nice", "-n", "15"] + command  # keep health checks responsive

        logger.info("Encoding video (%d slides, %.1fs)...", len(durations), sum(durations))
        result = subprocess.run(
            command, cwd=workdir, capture_output=True, text=True, timeout=900
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")

    return sum(durations)
