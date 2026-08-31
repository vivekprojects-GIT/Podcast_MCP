# Podcast MCP

A lightweight MCP server that turns a podcast **script** into a finished **MP3**,
designed to run on **Render Free**. Three CPU-only TTS engines, switched with `TTS_ENGINE`:

- **`piper`** (Render default) — [Piper TTS](https://github.com/OHF-Voice/piper1-gpl)
  with rhasspy voices (~63 MB each, GPL-licensed engine). Clear voices and the only
  engine that fits the free 512MB instance under real load (~230 MB steady, ~370 MB
  peak, measured).
- **`kitten`** (local default) — [KittenTTS Nano](https://github.com/KittenML/KittenTTS).
  Small download, but onnxruntime peaks at ~550 MB during generation — gets
  OOM-killed on Render Free.
- **`kokoro`** (opt-in) — [Kokoro-82M](https://github.com/thewh1teagle/kokoro-onnx)
  via ONNX. The most natural voices; needs a paid instance.

The reasoning stays in your main app; this service only does audio:

```text
Report App (its own LLM)
   ↓  report → HOST/GUEST dialogue script
Podcast MCP on Render
   ↓  1. parse script into speaker turns
   ↓  2. split turns into TTS-safe chunks
   ↓  3. KittenTTS generates host + guest audio
   ↓  4. merge with natural pauses → MP3
   ↓  5. serve file at /audio/<name>.mp3
returns audio_url
   ↓
Report App shows audio player
```

## Endpoints

| Path | What |
|---|---|
| `POST /mcp` | MCP streamable-HTTP endpoint (stateless, JSON responses) |
| `GET /health` | Health check (used by Render) |
| `GET /audio/{filename}` | Serves generated MP3/WAV files |

## MCP tools

### `generate_podcast_from_script`

```python
generate_podcast_from_script(
    script: str,            # "HOST: ...\nGUEST: ..." (any speaker labels work)
    title: str = "",
    host_voice: str = "",   # empty = engine default (piper: en_US-hfc_male-medium)
    guest_voice: str = "",  # empty = engine default (piper: en_US-hfc_female-medium)
    speed: float = 1.0,
)
```

Returns:

```json
{
  "success": true,
  "type": "podcast",
  "title": "Q2 Business Review",
  "audio_url": "https://podcast-mcp.onrender.com/audio/q2-business-review-a1b2c3d4.mp3",
  "duration_seconds": 312.4,
  "turns": 14,
  "voices": {"HOST": "Jasper", "GUEST": "Bella"}
}
```

Script format (markdown decoration and `[cues]` are tolerated; unlabeled lines
continue the previous speaker):

```text
HOST: Welcome back to the show. Today we're looking at the Q2 results.
GUEST: Thanks for having me. The headline: revenue grew 18 percent.
HOST: Let's break that down...
```

### `text_to_speech`

```python
text_to_speech(text: str, voice: str = "", speed: float = 1.0, format: str = "mp3")
```

### `list_voices`

Returns the active engine, its voices, and the current defaults.

- **piper**: `en_US-hfc_male-medium` (default host), `en_US-hfc_female-medium`
  (default guest), plus ryan/amy/lessac/joe/kristin/kusal and `en_GB` voices; any
  voice name from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
  works and is downloaded on demand.

### Languages (piper engine)

Piper voices are **natively trained per language** (~35 languages in the rhasspy
catalog) — this is real multilingual speech, not phonemization tricks. Write the
script in the target language and pick voices whose locale matches:

| Language | Example host / guest voices |
|---|---|
| English | `en_US-hfc_male-medium` / `en_US-hfc_female-medium` |
| Dutch | `nl_NL-mls-medium` / `nl_BE-nathalie-medium` |
| German | `de_DE-thorsten-medium` |
| French / Spanish | `fr_FR-siwis-medium` / `es_ES-davefx-medium` |
| Hindi | `hi_IN-rohan-medium` / `hi_IN-priyamvada-medium` |
| Telugu | `te_IN-venkatesh-medium` / `te_IN-maya-medium` |

A voice speaks only its own language — don't mix an `en_US` voice with a Dutch
script. Note: the kitten engine is English-only (espeak can *phonemize* 100+
languages, but the acoustic model is trained on English, so other languages
come out garbled — don't advertise it as multilingual).
- **kokoro**: 27 English voices — `af_*`/`am_*` American female/male, `bf_*`/`bm_*` British
  (e.g. `af_heart`, `af_bella`, `am_michael`, `am_adam`, `bf_emma`, `bm_george`).
- **kitten**: `Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo`.

## Calling it from your report app

With the official Python MCP client:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def make_podcast(script: str, title: str) -> str:
    async with streamablehttp_client("https://podcast-mcp.onrender.com/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "generate_podcast_from_script",
                {"script": script, "title": title},
            )
            return result.structuredContent["audio_url"]
```

Or add it to any MCP-capable agent as a remote server with URL
`https://<your-service>.onrender.com/mcp`.

## Deploy on Render (free)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo — [render.yaml](render.yaml) provisions a free
   Docker web service with `/health` checks.
3. Done. `RENDER_EXTERNAL_URL` is used automatically to build `audio_url`s
   (override with `PUBLIC_BASE_URL` if you put a domain in front).

Notes for the free tier:

- First boot downloads the two default piper voices (~126 MB) into `/tmp` in a
  background preload, so the service is healthy immediately; the first tool call
  may wait on it.
- The instance sleeps after idle; the first request after a sleep takes ~1 min plus the
  model re-download (the disk is wiped on sleep/restart).
- Keep `TTS_ENGINE=piper` on the free instance — kitten and kokoro exceed 512 MB
  under load and get OOM-killed (502s mid-generation). They work on paid instances.
- Audio files live on ephemeral disk and are deleted after `AUDIO_TTL_HOURS` (24h default)
  or on restart — have your app fetch/cache the MP3 promptly if it must keep it.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `TTS_ENGINE` | `kitten` (render.yaml sets `piper`) | `piper` (free-tier safe), `kitten`, or `kokoro` (needs paid RAM) |
| `LOW_MEMORY_MODE` | `1` | Disables onnxruntime's memory arena + caps threads. Set `0` on big instances for speed |
| `TTS_PACE` | `0.6` | Duty-cycle pacing: per-chunk sleep = chunk time × this, so health checks stay alive on 0.1-CPU instances. Set `0` on real CPUs |
| `PIPER_MODEL_DIR` | `models/piper` (`/tmp/piper` in Docker) | Where Piper voice files are cached |
| `KOKORO_VARIANT` | `int8` | `int8` (114 MB), `fp16` (164 MB), or `fp32` (326 MB) |
| `KOKORO_MODEL_DIR` | `models/kokoro` (`/tmp/kokoro` in Docker) | Where Kokoro model files are cached |
| `KITTEN_MODEL` | `KittenML/kitten-tts-nano-0.8` | Full-precision nano (~56MB). The `-int8` variant is smaller but has known quality issues |
| `DEFAULT_HOST_VOICE` / `DEFAULT_GUEST_VOICE` | engine defaults | Override default voices |
| `PUBLIC_BASE_URL` | `RENDER_EXTERNAL_URL` or localhost | Base for returned `audio_url` |
| `AUDIO_DIR` | `audio_output` (`/tmp/podcast_audio` in Docker) | Where files are written |
| `AUDIO_TTL_HOURS` | `24` | Delete generated files older than this |
| `MAX_SCRIPT_CHARS` | `20000` | Reject oversized scripts |
| `PRELOAD_MODEL` | `1` | Load the model in the background at boot |

## Run locally

Works with plain pip on Windows/Mac/Linux (KittenTTS 0.8.1 bundles espeak via
`espeakng-loader`, no system packages needed):

```bash
pip install -r requirements.txt
```

```bash
python server.py
```

Then the MCP endpoint is `http://localhost:8000/mcp`. Or with Docker (same image Render uses):

```bash
docker build -t podcast-mcp .
```

```bash
docker run -p 8000:8000 podcast-mcp
```

Pure-logic tests (no model needed):

```bash
python test_logic.py
```
