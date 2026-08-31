FROM python:3.12-slim

# espeak-ng: phonemization backend used by some KittenTTS builds
# ffmpeg + fonts-dejavu-core: slide-video rendering (generate_video_from_sections)
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render Free has no persistent disk; audio + model caches live in /tmp
ENV AUDIO_DIR=/tmp/podcast_audio \
    HF_HOME=/tmp/hf_cache \
    KOKORO_MODEL_DIR=/tmp/kokoro \
    PIPER_MODEL_DIR=/tmp/piper \
    PORT=8000

# Free tier = 512MB RAM / 0.1 CPU: single-threaded math kernels avoid thread
# thrash, and capped glibc arenas keep RSS from creeping past the memory limit
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

EXPOSE 8000

CMD ["python", "server.py"]
