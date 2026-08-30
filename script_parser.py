"""Parse an LLM-written podcast script into (speaker, text) turns.

Accepted line formats (label is case-insensitive, markdown decoration tolerated):
    HOST: Welcome to the show.
    **Guest:** Thanks for having me.
    [Sarah]: Let's dig into the numbers.
    Unlabeled lines continue the previous speaker's turn.
Lines that are only a stage direction, e.g. "[intro music]", are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Speaker label: 1-3 words before a colon, e.g. "HOST", "Speaker 1", "Dr. Chen"
_LABEL_RE = re.compile(
    r"^\s*[*_\[\(>#-]*\s*([A-Za-z][A-Za-z0-9.'-]*(?:\s+[A-Za-z0-9.'-]+){0,2})\s*[*_\]\)]*\s*:\s*(.*)$"
)
_STAGE_DIRECTION_RE = re.compile(r"^\s*[\[\(][^\]\)]*[\]\)]\s*$")
_INLINE_CUE_RE = re.compile(r"\[[^\]]*\]")  # [laughs], [pause] ...

_HOST_ALIASES = {"HOST", "HOST 1", "A", "S1", "SPEAKER 1", "SPEAKER A",
                 "ANCHOR", "INTERVIEWER", "MODERATOR", "NARRATOR"}
_GUEST_ALIASES = {"GUEST", "B", "S2", "SPEAKER 2", "SPEAKER B",
                  "EXPERT", "INTERVIEWEE", "ANALYST"}


@dataclass
class Turn:
    speaker: str  # normalized: "HOST", "GUEST", or the raw name uppercased
    text: str


def normalize_speaker(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().upper())
    if key in _HOST_ALIASES:
        return "HOST"
    if key in _GUEST_ALIASES:
        return "GUEST"
    return key


def _clean_text(text: str) -> str:
    text = _INLINE_CUE_RE.sub(" ", text)
    text = text.replace("*", " ").replace("`", " ").replace("#", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_script(script: str) -> list[Turn]:
    turns: list[Turn] = []
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or _STAGE_DIRECTION_RE.match(line):
            continue
        match = _LABEL_RE.match(line)
        if match:
            speaker = normalize_speaker(match.group(1))
            text = _clean_text(match.group(2))
        elif turns:
            speaker = turns[-1].speaker
            text = _clean_text(line)
        else:
            speaker, text = "HOST", _clean_text(line)
        if not text:
            continue
        if turns and turns[-1].speaker == speaker:
            turns[-1].text += " " + text
        else:
            turns.append(Turn(speaker, text))
    return turns


def split_into_chunks(text: str, max_chars: int = 300) -> list[str]:
    """Split text into TTS-safe chunks on sentence boundaries.

    KittenTTS quality degrades on very long inputs, so each chunk is kept
    under max_chars. Oversized sentences are further split on commas/spaces.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    parts = re.split(r",\s*", sentence)
    out: list[str] = []
    current = ""
    for part in parts:
        while len(part) > max_chars:  # pathological: no commas either
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        if current and len(current) + 2 + len(part) > max_chars:
            out.append(current)
            current = part
        else:
            current = f"{current}, {part}".strip(", ").strip()
    if current:
        out.append(current)
    return out
