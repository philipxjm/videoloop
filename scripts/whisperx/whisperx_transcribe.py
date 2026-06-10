#!/usr/bin/env python3
"""Batch transcribe all videos using WhisperX with speaker diarization.

Produces enhanced transcripts in dataset/videomme_long/transcripts_whisperx/{video_id}.json
with word-level timestamps and speaker labels.

Usage:
    # Full run (all 300 videos)
    python scripts/whisperx_transcribe.py --video-dir dataset/videomme_long/videos

    # Resume (skip already-done videos)
    python scripts/whisperx_transcribe.py --video-dir dataset/videomme_long/videos --resume

    # Skip diarization (if pyannote license not accepted yet)
    python scripts/whisperx_transcribe.py --video-dir dataset/videomme_long/videos --no-diarize
"""

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import whisperx
from tqdm import tqdm

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = 8
WHISPER_MODEL = "large-v3"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "dataset" / "videomme_long" / "transcripts_whisperx"


def transcribe_video(
    video_path: str,
    whisper_model,
    align_models: dict,
    diarize_pipeline,
) -> dict:
    """Transcribe a single video with WhisperX.

    Returns dict with keys: segments, language, duration_s
    Each segment has: start, end, text, speaker (if diarized)
    """
    audio = whisperx.load_audio(video_path)
    duration_s = len(audio) / 16000

    # 1. Transcribe
    result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
    language = result.get("language", "en")

    # 2. Align (word-level timestamps)
    if language not in align_models:
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )
            align_models[language] = (model_a, metadata)
        except Exception as e:
            print(f"    Warning: no alignment model for language '{language}': {e}")
            align_models[language] = None

    if align_models.get(language) is not None:
        model_a, metadata = align_models[language]
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=DEVICE,
            return_char_alignments=False,
        )

    # 3. Diarize (speaker labels) — try GPU, fall back to CPU on OOM
    if diarize_pipeline is not None:
        try:
            diarize_df = diarize_pipeline(audio)
            result = whisperx.assign_word_speakers(diarize_df, result)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                tqdm.write(f"    GPU OOM during diarization, falling back to CPU...")
                gc.collect()
                torch.cuda.empty_cache()
                diarize_pipeline.model.to(torch.device("cpu"))
                diarize_df = diarize_pipeline(audio)
                result = whisperx.assign_word_speakers(diarize_df, result)
                diarize_pipeline.model.to(torch.device(DEVICE))
            else:
                tqdm.write(f"    Warning: diarization failed: {e}")
        except Exception as e:
            tqdm.write(f"    Warning: diarization failed: {e}")

    # 4. Format output
    segments = []
    for seg in result["segments"]:
        entry = {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        if "speaker" in seg:
            entry["speaker"] = seg["speaker"]
        segments.append(entry)

    return {
        "segments": segments,
        "language": language,
        "duration_s": round(duration_s, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch WhisperX transcription")
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for transcripts",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-done videos")
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization",
    )
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help="Whisper model size (default: large-v3)",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect video files
    videos = sorted(
        [f for f in video_dir.iterdir() if f.suffix.lower() in (".mp4", ".mkv", ".avi", ".webm")]
    )
    print(f"Found {len(videos)} videos in {video_dir}")

    # Filter already-done if resuming
    if args.resume:
        remaining = []
        for v in videos:
            vid_id = v.stem
            out_path = output_dir / f"{vid_id}.json"
            if not out_path.exists():
                remaining.append(v)
        print(f"Resuming: {len(videos) - len(remaining)} already done, {len(remaining)} remaining")
        videos = remaining

    if not videos:
        print("Nothing to do!")
        return

    # Load whisper model
    print(f"Loading WhisperX model ({args.model})...")
    whisper_model = whisperx.load_model(
        args.model, device=DEVICE, compute_type=COMPUTE_TYPE
    )
    print("WhisperX model loaded")

    # Alignment models cache (loaded per-language on demand)
    align_models: dict = {}

    # Load diarization pipeline
    diarize_pipeline = None
    if not args.no_diarize:
        if not HF_TOKEN:
            print("Warning: HF_TOKEN not set, skipping diarization")
        else:
            try:
                from whisperx.diarize import DiarizationPipeline

                print("Loading diarization pipeline...")
                diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-3.1",
                    token=HF_TOKEN,
                    device=DEVICE,
                )
                print(f"Diarization pipeline loaded (on {DEVICE})")
            except Exception as e:
                print(f"Warning: could not load diarization pipeline: {e}")
                print("Proceeding without diarization")

    # Process videos
    success = 0
    failed = []

    pbar = tqdm(videos, desc="Transcribing", unit="video")
    for video_path in pbar:
        vid_id = video_path.stem
        out_path = output_dir / f"{vid_id}.json"
        pbar.set_postfix_str(vid_id, refresh=True)

        try:
            t0 = time.time()
            result = transcribe_video(
                str(video_path), whisper_model, align_models, diarize_pipeline
            )
            dt = time.time() - t0

            # Save
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            n_seg = len(result["segments"])
            n_speakers = len(
                set(s.get("speaker", "") for s in result["segments"] if s.get("speaker"))
            )
            tqdm.write(
                f"  {vid_id}: {n_seg} segments, {n_speakers} speakers, "
                f"lang={result['language']}, {result['duration_s']}s audio, "
                f"{dt:.1f}s processing"
            )
            success += 1

        except Exception as e:
            tqdm.write(f"  {vid_id} FAILED: {e}")
            traceback.print_exc()
            failed.append(vid_id)

        # Free memory after every video
        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(videos)} videos transcribed")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
