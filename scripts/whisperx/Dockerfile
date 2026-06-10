FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install whisperx first (pulls CPU torch as a side-effect), then
# reinstall torch/torchaudio with CUDA 12.8 support.
RUN pip install --no-cache-dir whisperx && \
    pip install --no-cache-dir --force-reinstall \
        torch~=2.8.0 torchaudio~=2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

COPY whisperx_transcribe.py /workspace/whisperx_transcribe.py

ENTRYPOINT ["python", "/workspace/whisperx_transcribe.py"]
