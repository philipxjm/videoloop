# WhisperX Transcription for VideoMME

Batch-transcribe VideoMME Long videos using [WhisperX](https://github.com/m-bain/whisperX) (large-v3) with word-level timestamps and speaker diarization.

## What It Does

The script (`whisperx_transcribe.py`) processes every `.mp4`/`.mkv`/`.avi`/`.webm` file in a video directory and produces one JSON transcript per video in the output directory. Each transcript contains:

```json
{
  "segments": [
    {
      "start": 6.31,
      "end": 8.63,
      "text": "Ah, it's blinding me.",
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "en",
  "duration_s": 2256.0
}
```

The pipeline runs three stages per video:
1. **Transcribe** - Whisper large-v3 via CTranslate2 (float16 on GPU, int8 on CPU)
2. **Align** - Word-level timestamp alignment (model loaded per-language on demand)
3. **Diarize** - Speaker labels via pyannote/speaker-diarization-3.1 (optional, requires HuggingFace token)

## Contents

```
scripts/whisperx/
├── Dockerfile
├── docker-compose.yml
├── README.md
└── whisperx_transcribe.py
```

## Prerequisites

- **NVIDIA GPU** with at least 8 GB VRAM (large-v3 + diarization). Works on CPU but will be very slow.
- **NVIDIA drivers** installed on the host (>=525.x recommended)
- **Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- **HuggingFace token** (optional, for speaker diarization) - you must accept the license agreements for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) on the HuggingFace Hub

## Quick Start (Docker Compose)

This is the easiest way to run on a new machine.

### 1. Enter the directory

```bash
cd scripts/whisperx
```

### 2. Set your HuggingFace token (optional, for speaker diarization)

```bash
export HF_TOKEN="hf_your_token_here"
```

If you skip this, the script will still transcribe but without speaker labels.

### 3. Point to your videos

Set `VIDEO_DIR` to whatever directory contains your `.mp4` files, and `OUTPUT_DIR` to where you want transcripts written:

```bash
export VIDEO_DIR=/path/to/your/videos
export OUTPUT_DIR=/path/to/output
```

If you don't set these, compose defaults to `./videos` and `./output` in the current directory.

### 4. Build and run

```bash
docker compose up --build
```

The `--resume` flag is on by default, so re-running will skip already-transcribed videos.

### 5. Stop

```bash
docker compose down
```

## Running with Docker Directly

```bash
# Build the image
docker build -t whisperx-transcribe .

# Run (adjust paths as needed)
docker run --gpus all \
  -e HF_TOKEN="hf_your_token_here" \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  whisperx-transcribe \
  --video-dir /data/videos \
  --output-dir /data/output \
  --resume
```

## Running Without Docker

If you prefer a local install (requires CUDA toolkit on the host):

```bash
# Create a venv
python3 -m venv .venv
source .venv/bin/activate

# Install whisperx, then reinstall torch with CUDA support
pip install whisperx
pip install --force-reinstall torch~=2.8.0 torchaudio~=2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Also need ffmpeg on the system
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# Run
export HF_TOKEN="hf_your_token_here"
python whisperx_transcribe.py \
  --video-dir /path/to/videos \
  --output-dir /path/to/output \
  --resume
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video-dir` | *(required)* | Directory containing video files |
| `--output-dir` | `dataset/videomme_long/transcripts_whisperx` | Where to write JSON transcripts |
| `--resume` | off | Skip videos that already have a transcript |
| `--no-diarize` | off | Skip speaker diarization (faster, no HF token needed) |
| `--model` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) |

## Performance

On an A100 (80GB), the full 300-video VideoMME Long set takes roughly 4-6 hours with diarization enabled. A single 30-minute video typically processes in 2-4 minutes.

For GPUs with less VRAM (e.g., RTX 3090 24GB), the script automatically falls back to CPU for diarization if GPU OOM occurs during that stage.

## Troubleshooting

**"CUDA not available" inside Docker**
- Verify the NVIDIA Container Toolkit is installed: `nvidia-smi` should work inside the container
- Make sure you're using `--gpus all` or the `deploy.resources` block in compose

**"HF_TOKEN not set, skipping diarization"**
- Export `HF_TOKEN` before running. For compose, it reads from your shell environment.

**"Could not load diarization pipeline"**
- Accept the pyannote model licenses on HuggingFace Hub (links above)
- Verify your token has read access to gated models

**GPU OOM during diarization**
- This is handled automatically - the script falls back to CPU for that stage
- If transcription itself OOMs, try a smaller model: `--model medium`

**Slow on CPU**
- Expected. large-v3 on CPU is ~50x slower than GPU. Use `--model tiny` or `--model base` for testing.
