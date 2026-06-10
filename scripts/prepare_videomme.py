#!/usr/bin/env python3
"""Build the VideoMME-Long dataset JSON from HuggingFace.

Downloads the `lmms-lab/Video-MME` question set, filters to the Long split
(300 videos, 900 questions), and rebuilds the evaluation JSON in the
pipeline's schema:

    dataset/videomme_long/videomme_long_full.json

Videos are ordered by ASCII sort of YouTube id. Video-level metadata
(duration_minutes, file_size_mb, resolution) is probed with ffprobe when
the videos are present locally.

Usage:
    python scripts/prepare_videomme.py                    # JSON only
    python scripts/prepare_videomme.py --download-videos  # also fetch videos (large!)

Videos are distributed by the benchmark authors as zip archives on the HF
repo; the Long split alone is several hundred GB. Video-MME is licensed for
research use only — see https://huggingface.co/datasets/lmms-lab/Video-MME

Requires: pip install videoloop[datasets]  (datasets, huggingface-hub)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "dataset" / "videomme_long"


def probe_video(path: Path) -> dict:
    """Return duration/size/resolution metadata via ffprobe, or {} on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(r.stdout)
        fmt = info.get("format", {})
        dur = float(fmt.get("duration", 0))
        size_mb = round(int(fmt.get("size", 0)) / 1048576, 1)
        res = ""
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                res = f"{s.get('width')}x{s.get('height')}"
                break
        return {
            "duration_minutes": round(dur / 60, 1),
            "file_size_mb": size_mb,
            "resolution": res,
        }
    except Exception:
        return {}


def download_videos(out_dir: Path, long_ids: set):
    """Download the official video archives and extract Long-split videos."""
    from huggingface_hub import HfApi, hf_hub_download

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files("lmms-lab/Video-MME", repo_type="dataset")
    zips = sorted(f for f in files if f.endswith(".zip"))
    print(f"{len(zips)} archives on the HF repo; extracting only Long-split videos")
    for zip_name in zips:
        print(f"Downloading {zip_name} ...")
        zp = hf_hub_download("lmms-lab/Video-MME", zip_name, repo_type="dataset")
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                stem = Path(member).stem
                if not member.lower().endswith(".mp4") or stem not in long_ids:
                    continue
                target = videos_dir / f"{stem}.mp4"
                if target.exists():
                    continue
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--download-videos", action="store_true",
                    help="Also download and extract the Long-split videos (very large)")
    args = ap.parse_args()

    from datasets import load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lmms-lab/Video-MME ...")
    ds = load_dataset("lmms-lab/Video-MME", "videomme", split="test")
    rows = [r for r in ds if r["duration"] == "long"]
    print(f"{len(rows)} Long-split questions")

    by_video: dict = defaultdict(list)
    meta: dict = {}
    for r in rows:
        yid = r["videoID"]
        by_video[yid].append(r)
        meta[yid] = r

    if args.download_videos:
        download_videos(args.out_dir, set(by_video))

    entries = []
    for yid in sorted(by_video):
        r0 = meta[yid]
        video_path = args.out_dir / "videos" / f"{yid}.mp4"
        probed = probe_video(video_path) if video_path.exists() else {}
        questions = []
        for r in sorted(by_video[yid], key=lambda x: x["question_id"]):
            questions.append({
                "question_id": r["question_id"].lstrip("0") if r["question_id"][0] == "0" else r["question_id"],
                "task_type": r["task_type"],
                "question": r["question"],
                "options": list(r["options"]),
                "answer": r["answer"],
            })
        entries.append({
            "video_id": r0["video_id"].lstrip("0") or r0["video_id"],
            "duration": r0["duration"],
            "domain": r0["domain"],
            "sub_category": r0["sub_category"],
            "url": yid,
            "video_file": f"videos/{yid}.mp4",
            "questions": questions,
            **probed,
        })

    total_q = sum(len(e["questions"]) for e in entries)
    print(f"{len(entries)} videos, {total_q} questions")

    out_path = args.out_dir / "videomme_long_full.json"
    out_path.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
