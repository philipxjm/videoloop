#!/usr/bin/env python3
"""Build the VideoMMMU dataset JSONs from HuggingFace.

Downloads `lmms-lab/VideoMMMU` (Perception / Comprehension / Adaptation
tracks, 268 videos x 3 = 804 questions) and rebuilds the evaluation JSONs
in the pipeline's schema:

    dataset/videommmu/videommmu_full.json        (804 questions)
    dataset/videommmu/videommmu_adaptation.json  (268 questions)

Videos are grouped by YouTube id and ordered by ASCII sort of the id;
Adaptation question images are embedded as base64 (original bytes from the
HF parquet). Video-level metadata (duration, resolution) is probed with
ffprobe when the videos are present.

Usage:
    python scripts/prepare_videommmu.py                    # JSONs only
    python scripts/prepare_videommmu.py --download-videos  # also fetch videos (~25GB)

Requires: pip install videoloop[datasets]  (datasets, huggingface-hub)
VideoMMMU is distributed under its authors' license — see
https://huggingface.co/datasets/lmms-lab/VideoMMMU
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "dataset" / "videommmu"
TRACKS = ["Perception", "Comprehension", "Adaptation"]
TRACK_SUFFIX = {"Perception": "perc", "Comprehension": "comp", "Adaptation": "adap"}
VIDEO_ZIPS = [
    "Art.zip", "Business.zip", "Engineering.zip",
    "Humanities.zip", "Medicine.zip", "Science.zip",
]


def youtube_id(link: str) -> str:
    """Extract the video id from a YouTube URL."""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", link)
    if not m:
        raise ValueError(f"Cannot parse YouTube id from {link!r}")
    return m.group(1)


def domain_from_hf_id(hf_id: str) -> str:
    """validation_Architecture_and_Engineering_26 -> Architecture_and_Engineering."""
    m = re.match(r"(?:dev|validation|test|new)_(.+?)_\d+$", hf_id)
    return m.group(1) if m else hf_id


def letter_options(options: list) -> list:
    """Prefix bare option strings with 'A. ', 'B. ', ... (idempotent)."""
    out = []
    for i, opt in enumerate(options):
        opt = str(opt)
        if re.match(r"^[A-J][.)] ", opt):
            out.append(opt)
        else:
            out.append(f"{chr(65 + i)}. {opt}")
    return out


def probe_video(path: Path) -> dict:
    """Return duration/resolution metadata via ffprobe, or {} on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(r.stdout)
        dur = float(info.get("format", {}).get("duration", 0))
        res = ""
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                res = f"{s.get('width')}x{s.get('height')}"
                break
        return {
            "duration_seconds": round(dur, 2),
            "duration_minutes": round(dur / 60, 2),
            "resolution": res,
        }
    except Exception:
        return {}


def download_videos(out_dir: Path):
    """Download and extract the official video archives from HF."""
    from huggingface_hub import hf_hub_download

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for zip_name in VIDEO_ZIPS:
        print(f"Downloading {zip_name} ...")
        zp = hf_hub_download("lmms-lab/VideoMMMU", zip_name, repo_type="dataset")
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".mp4"):
                    continue
                target = videos_dir / Path(member).name
                if target.exists():
                    continue
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        print(f"  extracted into {videos_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--download-videos", action="store_true",
                    help="Also download and extract the video archives (~25GB)")
    args = ap.parse_args()

    from datasets import Image, load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.download_videos:
        download_videos(args.out_dir)

    # ── Load all three tracks, grouped by YouTube id ───────────────────
    by_video: dict = defaultdict(dict)
    for track in TRACKS:
        print(f"Loading {track} ...")
        ds = load_dataset("lmms-lab/VideoMMMU", track, split="test")
        if "image" in ds.column_names:
            # decode=False keeps the original encoded bytes for faithful embedding
            ds = ds.cast_column("image", Image(decode=False))
        for row in ds:
            yid = youtube_id(row["link_selected"])
            q = {
                "task_type": track,
                "question": row["question"],
                "options": letter_options(row["options"]),
                "answer": row["answer"],
                "hf_id": row["id"],
                "qa_type": row["qa_type"],
                "question_type": row["question_type"],
            }
            img = row.get("image")
            if img and img.get("bytes"):
                q["image_base64"] = base64.b64encode(img["bytes"]).decode("ascii")
            by_video[yid][track] = q

    # ── Assemble video entries, ordered by ASCII sort of YouTube id ────
    full, adaptation = [], []
    for idx, yid in enumerate(sorted(by_video), start=1):
        tracks = by_video[yid]
        missing = [t for t in TRACKS if t not in tracks]
        if missing:
            print(f"WARNING: {yid} missing tracks {missing}, skipping")
            continue
        perc = tracks["Perception"]
        meta = {
            "video_id": str(idx),
            "youtube_id": yid,
            "domain": domain_from_hf_id(perc["hf_id"]),
            "sub_category": perc["qa_type"],
            "url": yid,
            "video_file": f"{yid}.mp4",
        }
        video_path = args.out_dir / "videos" / f"{yid}.mp4"
        probed = probe_video(video_path) if video_path.exists() else {}

        questions = []
        for track in TRACKS:
            q = dict(tracks[track])
            q = {"question_id": f"{idx}-{TRACK_SUFFIX[track]}", **q}
            questions.append(q)

        full.append({**meta, "questions": questions, **probed})
        adaptation.append({**meta, "questions": [questions[2]], **probed})

    n_full = sum(len(v["questions"]) for v in full)
    n_adap = sum(len(v["questions"]) for v in adaptation)
    print(f"{len(full)} videos | full: {n_full} questions | adaptation: {n_adap} questions")

    full_path = args.out_dir / "videommmu_full.json"
    adap_path = args.out_dir / "videommmu_adaptation.json"
    full_path.write_text(json.dumps(full, indent=2))
    adap_path.write_text(json.dumps(adaptation, indent=2))
    print(f"Wrote {full_path}\nWrote {adap_path}")


if __name__ == "__main__":
    main()
