<h1 align="center">VideoLoop: Looped Working Memory Against Semantic Thrashing in Long-Form Video Agents</h1>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b?logo=arxiv&logoColor=white" alt="arXiv">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0">
  </a>
</p>

<p align="center">
  <a href="#Highlights">Highlights</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#environment-setup">Environment Setup</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#repository-structure">Repository Structure</a> ·
  <a href="#acknowledgements">Acknowledgements</a> ·
  <a href="#license">License</a> ·
  <a href="#citation">Citation</a>
</p>

VideoLoop is a dual-loop multimodal agent for long-form video understanding. An outer loop reasons over the video and runs tools inside a Docker sandbox; after every step, an inner loop (the *memory orchestrator*) retrieves evidence from the sandbox filesystem and rewrites a bounded working memory, so that evidence stays dense instead of being diluted by an ever-growing, append-only context. This repository provides the agent, the evaluation pipeline, dataset preparation scripts, and a live evaluation dashboard.


<a id="Highlights"></a>
## Highlights ✨

<p align="center">
<img src="assets/figure1.png" width="90%">
</p>

| Direction | Description |
| --- | --- |
| Semantic Thrashing Analysis | Identify and formalize why append-only memory fails on long videos: it can add new evidence but never remove accumulated noise, so the agent loses access to what it has already found. |
| Dual-Loop Bounded Memory | Pair the reasoning loop with an inner memory orchestrator that retrieves from an unbounded sandbox filesystem and rewrites a bounded working memory after every step. |
| Plug-and-Play Gains | Training-free gains on four LVLM backbones (+4.2 points on average on VideoMME-Long) and new state-of-the-art results with Gemini 3.1 Pro: 88.3% on VideoMME-Long, 88.8% on VideoMMMU, and 80.9% on LongVideoBench-Long. |

## Table of Contents
- [Highlights ✨](#highlights-)
- [Table of Contents](#table-of-contents)
- [1. Overview 🧠](#1-overview-)
- [2. Results 📊](#2-results-)
- [3. Environment Setup 🛠️](#3-environment-setup-️)
- [4. Quick Start 🚀](#4-quick-start-)
  - [4.1 Prepare Datasets](#41-prepare-datasets)
  - [4.2 Run a Single Question](#42-run-a-single-question)
  - [4.3 Run the Full Evaluation](#43-run-the-full-evaluation)
  - [4.4 Results Format](#44-results-format)
  - [4.5 Configuration](#45-configuration)
- [5. Repository Structure 🗂️](#5-repository-structure-️)
- [6. Acknowledgements 🙏](#6-acknowledgements-)
- [7. License ⚖️](#7-license-️)
- [8. Citation 📚](#8-citation-)


<a id="overview"></a>
## 1. Overview 🧠

<p align="center">
<img src="assets/framework.png" width="95%">
</p>

| Component | Description |
| --- | --- |
| Outer loop | The policy model reasons over a bounded context (the question, the working memory, and the last 8 message groups) and calls `analyze_frames`, `transcribe_audio`, `execute_bash`, or `submit_answer`, for up to 50 iterations. |
| Memory orchestrator | After every step, it reads the new observation, retrieves question-relevant artifacts with `read_file` (and can look at stored frames with `inspect_frames`), and applies section-level update / append / delete edits to the working memory. |
| Working memory | A six-section document (metadata, narrative understanding, timestamped evidence, temporal coverage, activity log, open investigation targets), kept within a fixed size budget and at most 6 key frames. It starts empty. |
| Step manifest | A compressed log of all past actions and their parameters (timestamps, queries, executed code), giving a navigable history that links the working memory to the filesystem. |
| Sandbox filesystem | Keeps every extracted frame, analysis output, transcript, and script losslessly across iterations, preserving the raw material from which evidence can be recovered. |


<a id="results"></a>
## 2. Results 📊

| Method | VideoMME-Long | VideoMMMU | LongVideoBench-Long |
| --- | :---: | :---: | :---: |
| Gemini 3 Flash | 80.7 | 83.6 | 67.9 |
| + VideoLoop | 85.8 | 87.7 | 73.8 |
| Gemini 3.1 Pro | 83.8 | 84.6 | 77.7 |
| **+ VideoLoop** | **88.3** | **88.8** | **80.9** |

Accuracy (%). See the paper for comparisons with prior video agents, additional backbones, ablations, and the semantic-thrashing analysis.


<a id="environment-setup"></a>
## 3. Environment Setup 🛠️

Requirements:

- `Python 3.10+` and `Docker`: the agent runs every tool inside a Docker sandbox.
- NVIDIA GPU, drivers, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html): the sandbox image is CUDA-based and `DockerRuntime` requests a GPU by default. On a host without a GPU, set `sandbox.gpu: false` in `configs/config.yaml` to run the sandbox CPU-only (frame extraction works on CPU; only WhisperX transcript generation needs a GPU).
- `ffmpeg`, if you run the dataset preparation or pre-compression scripts on the host.

Create the Python environment:

```bash
conda create -n videoloop python=3.10 -y
conda activate videoloop
pip install -e ".[dashboard,datasets]"
```

Build the Docker sandbox:

```bash
cd docker
./build.sh base    # First time: builds base image (~10-15 min)
./build.sh tools   # Builds tools layer on top (~5 sec)
```

Verify GPU access in the built image (skip if running CPU-only):

```bash
docker run --rm -it --gpus all video-understanding-sandbox:latest nvidia-smi
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

With no `api_base` configured, all components call the official Gemini API. Any OpenAI-compatible endpoint can be substituted via `api_base` in `configs/config.yaml` (or `MAIN_AGENT_API_BASE`).

<a id="quick-start"></a>
## 4. Quick Start 🚀

### 4.1 Prepare Datasets

The benchmark JSONs are rebuilt from the official HuggingFace releases; nothing is redistributed in this repository. Both datasets are gated on the Hub, so first accept each dataset's terms on its HF page, then either run `huggingface-cli login` or set `HF_TOKEN` in `.env`:

```bash
python scripts/prepare_videommmu.py --download-videos   # ~25GB of videos
python scripts/prepare_videomme.py  --download-videos   # Long split, several hundred GB
```

- [VideoMMMU](https://huggingface.co/datasets/lmms-lab/VideoMMMU): `dataset/videommmu`
- [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME): `dataset/videomme_long`

Both scripts also work without `--download-videos` if you obtain the videos separately (place them in `dataset/<benchmark>/videos/`). Video-MME is licensed for research use only; VideoMMMU is under its authors' license. See the respective HF dataset cards.

Transcripts are read from cached WhisperX outputs in `dataset/<benchmark>/transcripts_whisperx/`. Generate them on a GPU machine with `scripts/whisperx/whisperx_transcribe.py`; see [scripts/whisperx/README.md](scripts/whisperx/README.md) for the Docker setup and CLI options. For very large videos, optional pre-compression is available via `python scripts/precompress_videos.py --help`.

### 4.2 Run a Single Question

```bash
python scripts/agent_cli.py single \
  --video dataset/videomme_long/videos/<id>.mp4 \
  --question "What color is the presenter's shirt?" \
  --options "A. Red,B. Blue,C. Green,D. White"
```

### 4.3 Run the Full Evaluation

```bash
python scripts/agent_cli.py parallel-eval \
  --dataset dataset/videomme_long/videomme_long_full.json \
  --video-dir dataset/videomme_long/videos \
  --workers 32
```

A live dashboard starts automatically at `http://localhost:8080` (disable it with `--no-dashboard`; see [dashboard/README.md](dashboard/README.md)).

Expected cost: the full configuration uses about 618K tokens per question on VideoMME-Long (about 559K input and 59K output, mostly frame images), measured with Gemini 3 Flash. Wall-clock time and dollar cost scale with the chosen model (`gemini-3.1-pro-preview` is slower and pricier than the Flash backbones) and the number of workers.

### 4.4 Results Format

Each run writes `logs/parallel_eval_<run-id>.json`:

```jsonc
{
  "summary": {
    "total_videos": 300, "total_questions": 900,
    "answered": 895, "failed": 5, "correct": 790,
    "accuracy": 0.883, "elapsed_seconds": 12345.6,
    "rate_seconds_per_question": 13.8,
    "num_workers": 32, "model": "gemini-3.1-pro-preview"
  },
  "results": [
    {
      "question_id": "743-1", "video_id": "...",
      "question": "...", "options": ["A. ...", "B. ..."],
      "expected": "C", "predicted": "C", "correct": true,
      "reasoning": "...",            // model's submitted justification
      "elapsed_seconds": 41.2,
      "trajectory": [                 // ordered agent steps
        {"step_type": "tool_use", "content": {...}, "timestamp": 0.0}
      ],
      "token_usage": {"total": {...}, "by_agent": {...}},
      "error": null                   // non-null string if the question failed
    }
  ]
}
```

The dashboard renders this live during the run; failed questions have `predicted: null` and a populated `error`.

### 4.5 Configuration

`configs/config.yaml` ships with the configuration used for the paper results (native multimodal agent, orchestrated memory, `analyze_frames`-only visual analysis, Gemini 3.1 Pro for both loops). Point `CONFIG_FILE` at another file in `configs/` to experiment. Prompts live in `configs/prompts.yaml` (override with `PROMPTS_FILE`).

Paper settings and the corresponding keys in `configs/config.yaml`:

- Outer loop: at most 50 iterations (`--max-iterations`, default 50) and at least 6 before submitting (`min_iterations_before_submit: 6`), with a recent window of 8 message groups (`recent_messages_window: 8`)
- Working memory: at most 6 key frames (`max_key_frames: 6`)
- Step manifest: the oldest 10 records are folded into one summary (`manifest_compression_batch: 10`) once 30 unsummarized records accumulate (`manifest_max_detailed: 30`)

<a id="repository-structure"></a>
## 5. Repository Structure 🗂️

```text
videoloop/
├── video_agent/
│   ├── agent.py
│   ├── memory/
│   ├── tools.py
│   ├── tool_handlers.py
│   ├── llm_api.py
│   ├── api_calls.py
│   ├── parallel_eval.py
│   └── docker_runtime.py
├── scripts/
│   ├── agent_cli.py
│   ├── prepare_videomme.py
│   ├── prepare_videommmu.py
│   └── whisperx/
├── configs/
├── dashboard/
├── docker/
├── tests/
└── assets/
```

Core directories:

| Path | Description |
| --- | --- |
| `video_agent/agent.py` | Outer agent loop: reasoning, tool calls, and frame analysis |
| `video_agent/memory/orchestrator.py` | Inner loop: rewrites the bounded working memory after every step |
| `video_agent/tools.py`, `video_agent/tool_handlers.py` | Tool schemas and their implementations |
| `video_agent/llm_api.py`, `video_agent/api_calls.py` | Gemini-native and OpenAI-compatible transport, provider dispatch, and normalization |
| `video_agent/parallel_eval.py` | Multi-worker benchmark evaluation with the live dashboard |
| `video_agent/docker_runtime.py` | Sandbox lifecycle |
| `scripts/agent_cli.py` | Entry point for single questions and full evaluation |
| `scripts/prepare_videomme.py`, `scripts/prepare_videommmu.py` | Rebuild the benchmark JSONs from HuggingFace |
| `configs/config.yaml` | Paper configuration (models, memory, tool whitelist) |
| `dashboard/` | Real-time and archived run viewer (FastAPI + static) |

Tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

<a id="acknowledgements"></a>
## 6. Acknowledgements 🙏

This work builds on the [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME), [VideoMMMU](https://huggingface.co/datasets/lmms-lab/VideoMMMU), and [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) benchmarks, and on [WhisperX](https://github.com/m-bain/whisperX) for transcripts. We sincerely thank the authors and maintainers of these resources.

<a id="license"></a>
## 7. License ⚖️

This repository is released under `Apache-2.0`. See `LICENSE` for the full license text. Benchmark data remains under the respective benchmark authors' licenses.

<a id="citation"></a>
## 8. Citation 📚

If you use this repository, please cite the corresponding paper:

```bibtex
@article{huang2026videoloop,
  title={VideoLoop: Looped Working Memory Against Semantic Thrashing in Long-Form Video Agents},
  author={Huang, Jinfa and Xu, Jianming and Lin, Jingyang and Yang, Zhengyuan and Luo, Jiebo},
  journal={arXiv preprint},
  year={2026}
}
```
