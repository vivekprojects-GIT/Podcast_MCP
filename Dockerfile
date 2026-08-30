FROM python:3.12-slim

# espeak-ng: phonemization backend used by some KittenTTS builds; small and safe
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render Free has no persistent disk; audio + model caches live in /tmp
ENV AUDIO_DIR=/tmp/podcast_audio \
    HF_HOME=/tmp/hf_cache \
    KOKORO_MODEL_DIR=/tmp/kokoro \
    PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
