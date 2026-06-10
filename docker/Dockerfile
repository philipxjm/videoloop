# Video Understanding Agent Sandbox
# Built on top of llm-in-sandbox pattern
# Comprehensive video processing and analysis tools for long video understanding

# Use devel image which includes libnvcuvid for GPU video decoding
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install NVIDIA Video Codec SDK (libnvcuvid for GPU video decoding)
RUN apt-get update -y && \
    apt-get install -y libnvidia-decode-530 libnvidia-encode-530 || true && \
    rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    # Core tools
    git curl wget build-essential ca-certificates software-properties-common \
    # Python
    python3.11 python3.11-venv python3.11-dev python3-pip \
    # Video/Audio processing
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    libatlas-base-dev gfortran \
    # Audio
    libsndfile1 portaudio19-dev sox libsox-fmt-all \
    # Image processing
    libopencv-dev tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    # Additional libs
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    # Useful utilities
    vim nano htop tree jq unzip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Create working directories
RUN mkdir -p /testbed /videos /outputs /models

# ==============================================================================
# VIDEO PROCESSING LIBRARIES
# ==============================================================================

# Core video/image processing
RUN pip install --no-cache-dir \
    opencv-python-headless==4.9.0.80 \
    decord==0.6.0 \
    imageio==2.34.0 \
    imageio-ffmpeg==0.4.9 \
    pillow==10.2.0 \
    scikit-image==0.22.0

# Scene detection
RUN pip install --no-cache-dir \
    scenedetect[opencv]==0.6.4

# Video utilities
RUN pip install --no-cache-dir \
    moviepy==1.0.3 \
    vidgear==0.3.3

# ==============================================================================
# AUDIO PROCESSING & ASR
# ==============================================================================

RUN pip install --no-cache-dir \
    openai-whisper==20231117 \
    pydub==0.25.1 \
    librosa==0.10.1 \
    soundfile==0.12.1

# ==============================================================================
# OCR & TEXT EXTRACTION
# ==============================================================================

RUN pip install --no-cache-dir \
    easyocr==1.7.1 \
    pytesseract==0.3.10 \
    pdf2image==1.17.0

# ==============================================================================
# OBJECT DETECTION & COMPUTER VISION
# ==============================================================================

RUN pip install --no-cache-dir \
    ultralytics==8.1.0 \
    supervision==0.19.0

# Install torch with CUDA (for YOLO, Whisper, etc.)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ==============================================================================
# MACHINE LEARNING & DEEP LEARNING
# ==============================================================================

RUN pip install --no-cache-dir \
    transformers==4.38.0 \
    accelerate==0.27.0 \
    bitsandbytes==0.42.0 \
    sentencepiece==0.2.0 \
    einops==0.7.0

# ==============================================================================
# DATA PROCESSING & UTILITIES
# ==============================================================================

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.0 \
    scipy==1.12.0 \
    matplotlib==3.8.3 \
    seaborn==0.13.2 \
    tqdm==4.66.2 \
    rich==13.7.0 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.1 \
    httpx==0.27.0 \
    aiohttp==3.9.3 \
    requests==2.31.0

# ==============================================================================
# LLM/API INTEGRATION
# ==============================================================================

RUN pip install --no-cache-dir \
    openai==1.12.0 \
    anthropic==0.18.0 \
    litellm==1.28.0

# ==============================================================================
# DOWNLOAD MODELS (cached in image)
# ==============================================================================

# Pre-download YOLO model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"

# Pre-download Whisper tiny model (larger models downloaded on demand)
RUN python -c "import whisper; whisper.load_model('tiny')"

# Pre-download EasyOCR models
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, download_enabled=True)"

# ==============================================================================
# HELPER SCRIPTS
# ==============================================================================

# Create video analysis helper script
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import json\n\
from pathlib import Path\n\
\n\
def get_video_info(video_path):\n\
    """Get video metadata using ffprobe"""\n\
    import subprocess\n\
    cmd = [\n\
        "ffprobe", "-v", "quiet",\n\
        "-print_format", "json",\n\
        "-show_format", "-show_streams",\n\
        video_path\n\
    ]\n\
    result = subprocess.run(cmd, capture_output=True, text=True)\n\
    return json.loads(result.stdout)\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) < 2:\n\
        print("Usage: video_info.py <video_path>")\n\
        sys.exit(1)\n\
    info = get_video_info(sys.argv[1])\n\
    print(json.dumps(info, indent=2))\n\
' > /usr/local/bin/video_info.py && chmod +x /usr/local/bin/video_info.py

# Create frame extraction helper with ffmpeg fallback
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
import subprocess\n\
import json\n\
from PIL import Image\n\
\n\
def get_video_duration(video_path):\n\
    """Get video duration using ffprobe"""\n\
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]\n\
    result = subprocess.run(cmd, capture_output=True, text=True)\n\
    if result.returncode == 0:\n\
        info = json.loads(result.stdout)\n\
        return float(info.get("format", {}).get("duration", 0))\n\
    return 0\n\
\n\
def extract_frames_ffmpeg(video_path, output_dir, num_frames=8, max_size=512):\n\
    """Extract frames using ffmpeg (more reliable)"""\n\
    os.makedirs(output_dir, exist_ok=True)\n\
    duration = get_video_duration(video_path)\n\
    if duration <= 0:\n\
        return []\n\
    \n\
    indices = []\n\
    for i in range(num_frames):\n\
        timestamp = (i * duration) / num_frames\n\
        output_path = f"{output_dir}/frame_{i:03d}.jpg"\n\
        cmd = [\n\
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,\n\
            "-vframes", "1", "-q:v", "2",\n\
            "-vf", f"scale=w=min({max_size}\\,iw):h=-2",\n\
            output_path\n\
        ]\n\
        result = subprocess.run(cmd, capture_output=True, text=True)\n\
        if result.returncode == 0 and os.path.exists(output_path):\n\
            indices.append(i)\n\
            print(f"Saved frame {i+1}/{num_frames}: frame_{i:03d}.jpg")\n\
    return indices\n\
\n\
def extract_frames_decord(video_path, output_dir, num_frames=8, max_size=512):\n\
    """Extract frames using decord (fast but less compatible)"""\n\
    from decord import VideoReader, cpu\n\
    import numpy as np\n\
    \n\
    os.makedirs(output_dir, exist_ok=True)\n\
    vr = VideoReader(video_path, ctx=cpu(0))\n\
    total = len(vr)\n\
    indices = [int(i * total / num_frames) for i in range(num_frames)]\n\
    \n\
    for i, idx in enumerate(indices):\n\
        frame = vr[idx].asnumpy()\n\
        img = Image.fromarray(frame)\n\
        if max(img.size) > max_size:\n\
            ratio = max_size / max(img.size)\n\
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))\n\
        img.save(f"{output_dir}/frame_{i:03d}.jpg", quality=85)\n\
        print(f"Saved frame {i+1}/{num_frames}: frame_{i:03d}.jpg")\n\
    return indices\n\
\n\
def extract_frames(video_path, output_dir, num_frames=8, max_size=512):\n\
    """Extract frames with fallback"""\n\
    # Try decord first (faster), fall back to ffmpeg\n\
    try:\n\
        return extract_frames_decord(video_path, output_dir, num_frames, max_size)\n\
    except Exception as e:\n\
        print(f"Decord failed, using ffmpeg: {e}", file=sys.stderr)\n\
        return extract_frames_ffmpeg(video_path, output_dir, num_frames, max_size)\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) < 3:\n\
        print("Usage: extract_frames.py <video_path> <output_dir> [num_frames]")\n\
        sys.exit(1)\n\
    num = int(sys.argv[3]) if len(sys.argv) > 3 else 8\n\
    extract_frames(sys.argv[1], sys.argv[2], num)\n\
' > /usr/local/bin/extract_frames.py && chmod +x /usr/local/bin/extract_frames.py

# Create ASR helper
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
import json\n\
import warnings\n\
warnings.filterwarnings("ignore")\n\
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"\n\
# Redirect stderr to suppress progress bars\n\
import io\n\
stderr_backup = sys.stderr\n\
sys.stderr = io.StringIO()\n\
import whisper\n\
sys.stderr = stderr_backup\n\
\n\
def transcribe_video(video_path, model_size="base"):\n\
    """Transcribe audio from video using Whisper"""\n\
    # Suppress all output during model loading and transcription\n\
    old_stderr = sys.stderr\n\
    sys.stderr = io.StringIO()\n\
    try:\n\
        model = whisper.load_model(model_size)\n\
        result = model.transcribe(video_path, verbose=False)\n\
    finally:\n\
        sys.stderr = old_stderr\n\
    return result\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) < 2:\n\
        print(json.dumps({"error": "Usage: transcribe.py <video_path> [model_size]"}))\n\
        sys.exit(1)\n\
    model = sys.argv[2] if len(sys.argv) > 2 else "base"\n\
    try:\n\
        result = transcribe_video(sys.argv[1], model)\n\
        print(json.dumps({"text": result["text"], "segments": result["segments"]}))\n\
    except Exception as e:\n\
        print(json.dumps({"error": str(e)}))\n\
' > /usr/local/bin/transcribe.py && chmod +x /usr/local/bin/transcribe.py

# Create scene detection helper
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import json\n\
from scenedetect import detect, ContentDetector\n\
\n\
def detect_scenes(video_path, threshold=27.0):\n\
    """Detect scene changes in video"""\n\
    scenes = detect(video_path, ContentDetector(threshold=threshold))\n\
    return [{"start": s[0].get_seconds(), "end": s[1].get_seconds()} for s in scenes]\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) < 2:\n\
        print("Usage: detect_scenes.py <video_path> [threshold]")\n\
        sys.exit(1)\n\
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 27.0\n\
    scenes = detect_scenes(sys.argv[1], thresh)\n\
    print(json.dumps(scenes, indent=2))\n\
' > /usr/local/bin/detect_scenes.py && chmod +x /usr/local/bin/detect_scenes.py

# Create YOLO detection helper
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import json\n\
from ultralytics import YOLO\n\
from PIL import Image\n\
\n\
def detect_objects(image_path, conf_threshold=0.25):\n\
    """Detect objects in image using YOLOv8"""\n\
    model = YOLO("yolov8x.pt")\n\
    results = model(image_path, conf=conf_threshold, verbose=False)\n\
    detections = []\n\
    for r in results:\n\
        for box in r.boxes:\n\
            detections.append({\n\
                "class": r.names[int(box.cls)],\n\
                "confidence": float(box.conf),\n\
                "bbox": box.xyxy[0].tolist()\n\
            })\n\
    return detections\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) < 2:\n\
        print("Usage: detect_objects.py <image_path> [confidence]")\n\
        sys.exit(1)\n\
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25\n\
    detections = detect_objects(sys.argv[1], conf)\n\
    print(json.dumps(detections, indent=2))\n\
' > /usr/local/bin/detect_objects.py && chmod +x /usr/local/bin/detect_objects.py

# Create OCR helper
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import json\n\
import warnings\n\
warnings.filterwarnings("ignore")\n\
import easyocr\n\
\n\
reader = None\n\
\n\
def convert_bbox(bbox):\n\
    """Convert bbox to JSON-serializable format"""\n\
    return [[float(coord) for coord in point] for point in bbox]\n\
\n\
def extract_text(image_path):\n\
    """Extract text from image using EasyOCR"""\n\
    global reader\n\
    if reader is None:\n\
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)\n\
    results = reader.readtext(image_path)\n\
    return [{"text": str(r[1]), "confidence": float(r[2]), "bbox": convert_bbox(r[0])} for r in results]\n\
\n\
if __name__ == "__main__":\n\
    if len(sys.argv) < 2:\n\
        print(json.dumps({"error": "Usage: extract_text.py <image_path>"}))\n\
        sys.exit(1)\n\
    try:\n\
        results = extract_text(sys.argv[1])\n\
        print(json.dumps(results))\n\
    except Exception as e:\n\
        print(json.dumps({"error": str(e)}))\n\
        sys.exit(1)\n\
' > /usr/local/bin/extract_text.py && chmod +x /usr/local/bin/extract_text.py

# Set working directory
WORKDIR /testbed

# Environment
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["/bin/bash"]
