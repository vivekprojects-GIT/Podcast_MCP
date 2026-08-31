"""Render narrated slide videos (MP4) from structured report sections.

Each section becomes one slide (title, bullet points, optional bar chart —
drawn with PIL, no matplotlib) shown for the length of its narration, which is
synthesized with the active TTS engine. ffmpeg muxes stills + narration into
an MP4; still-image encoding keeps CPU and RAM small enough for Render Free.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tts import SAMPLE_RATE, TTSEngine, encode_wav
from script_parser import split_into_chunks

logger = logging.getLogger("podcast-mcp.video")

WIDTH, HEIGHT = 1280, 720
BG = (15, 23, 42)         # slate-900
FG = (241, 245, 249)      # slate-100
MUTED = (148, 163, 184)   # slate-400
ACCENT = (56, 189, 248)   # sky-400
BAR_ALT = (100, 116, 139)  # slate-500

SECTION_PAUSE = 0.7  # silence appended after each section's narration

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
                        HEIGHT - chart_y - 60)

    image.save(path, "PNG")


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
        waveforms: list = []
        durations: list[float] = []

        for index, section in enumerate(sections):
            slide_path = workdir / f"slide_{index:03d}.png"
            render_slide(section, index, len(sections), deck_title, slide_path)

            narration = _section_narration(section, index)
            chunks = split_into_chunks(narration, max_chunk_chars)
            waveform = engine.synthesize_turns([(voice, chunks)], speed=speed)
            waveform = np.concatenate([waveform, pause])
            waveforms.append(waveform)
            durations.append(len(waveform) / SAMPLE_RATE)
            logger.info("Section %d/%d: %.1fs narration", index + 1, len(sections),
                        durations[-1])

        narration_path = workdir / "narration.wav"
        encode_wav(np.concatenate(waveforms), narration_path)
        waveforms.clear()

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
