#!/usr/bin/env python3
"""
Pre-compress videos for evaluation, caching results locally.

GPU-accelerated (NVENC) with automatic CPU fallback.
  480p, 2fps, h264_nvenc/libx264, AAC 48kbps, target ≤85MB

Usage:
  python scripts/precompress_videos.py --video-dir dataset/videomme_long/videos --output-dir dataset/videomme_long/compressed --workers 32
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MAX_VIDEO_BYTES = 85 * 1024 * 1024  # 85 MB

# NVENC on consumer GPUs supports ~5 concurrent sessions.
# Semaphore prevents "too many sessions" errors.
_nvenc_semaphore = threading.Semaphore(5)
_has_nvenc: bool | None = None  # Lazy-detected


def _detect_nvenc() -> bool:
    """Check if h264_nvenc is available."""
    global _has_nvenc
    if _has_nvenc is not None:
        return _has_nvenc
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _has_nvenc = "h264_nvenc" in r.stdout
    except Exception:
        _has_nvenc = False
    return _has_nvenc


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def _build_encode_cmd(
    video_path: str, out_path: str, video_bps: int, audio_bps: int, use_gpu: bool,
) -> list:
    """Build ffmpeg command for either GPU or CPU encoding.

    GPU mode: cuvid hardware decode + software filter + NVENC encode.
    This avoids the hwupload_cuda filter chain issue and lets NVENC
    handle the encoding while the CPU does the lightweight fps/scale filtering.
    """
    vf_filters = "fps=2,scale=-2:480"

    if use_gpu:
        return [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",  # HW decode only, output to system memory
            "-i", video_path,
            "-vf", vf_filters,  # Software filter (fps + scale on CPU — trivial at 480p)
            "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", str(video_bps),
            "-maxrate", str(int(video_bps * 1.5)),
            "-bufsize", str(int(video_bps * 3)),
            "-c:a", "aac", "-b:a", str(audio_bps),
            "-movflags", "+faststart",
            out_path,
        ]
    else:
        return [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", vf_filters,
            "-c:v", "libx264", "-preset", "fast", "-b:v", str(video_bps),
            "-maxrate", str(int(video_bps * 1.5)),
            "-bufsize", str(int(video_bps * 3)),
            "-c:a", "aac", "-b:a", str(audio_bps),
            "-movflags", "+faststart",
            out_path,
        ]


def _run_encode(
    video_path: str, out_path: str, video_bps: int, audio_bps: int, use_gpu: bool,
) -> subprocess.CompletedProcess:
    """Run encode with GPU semaphore if needed, fallback to CPU on failure."""
    if use_gpu:
        with _nvenc_semaphore:
            cmd = _build_encode_cmd(video_path, out_path, video_bps, audio_bps, True)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0:
            return proc
        # NVENC failed (session limit, etc.) — fallback to CPU
        if os.path.exists(out_path):
            os.unlink(out_path)

    cmd = _build_encode_cmd(video_path, out_path, video_bps, audio_bps, False)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def compress_video(video_path: str, out_path: str, max_bytes: int, use_gpu: bool) -> dict:
    """Compress a single video. Returns status dict."""
    stem = Path(video_path).stem
    size = os.path.getsize(video_path)
    size_mb = size / 1024 / 1024

    if size <= max_bytes:
        # Always compress for consistency (same codec/fps/resolution)
        pass

    try:
        duration = get_video_duration(video_path)
    except Exception as e:
        return {"name": stem, "status": "error", "error": f"ffprobe failed: {e}"}

    audio_bps = 48_000
    target_total_bps = int(max_bytes * 8 * 0.95 / duration)
    video_bps = max(target_total_bps - audio_bps, 50_000)

    proc = _run_encode(video_path, out_path, video_bps, audio_bps, use_gpu)
    if proc.returncode != 0:
        return {"name": stem, "status": "error", "error": proc.stderr[-300:]}

    actual = os.path.getsize(out_path)

    # Second pass if still too big
    if actual > max_bytes:
        ratio = max_bytes / actual * 0.9
        video_bps2 = max(int(video_bps * ratio), 30_000)
        tmp_path = out_path + ".tmp.mp4"
        proc2 = _run_encode(video_path, tmp_path, video_bps2, 32_000, use_gpu)
        if proc2.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, out_path)
            actual = os.path.getsize(out_path)
        elif os.path.exists(tmp_path):
            os.unlink(tmp_path)

    final_mb = actual / 1024 / 1024
    return {"name": stem, "status": "compressed", "original_mb": size_mb, "final_mb": final_mb}


def process_one(video_path: str, output_dir: str, max_bytes: int, use_gpu: bool) -> dict:
    """Process a single video — skip if already cached."""
    stem = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{stem}.mp4")

    if os.path.exists(out_path) or os.path.islink(out_path):
        final_mb = os.path.getsize(out_path) / 1024 / 1024
        return {"name": stem, "status": "cached", "final_mb": final_mb}

    t0 = time.time()
    result = compress_video(video_path, out_path, max_bytes, use_gpu)
    result["time"] = time.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser(description="Pre-compress videos for evaluation (GPU-accelerated)")
    parser.add_argument("--video-dir", type=str, default="dataset/videomme_long/videos", help="Source video directory")
    parser.add_argument("--output-dir", type=str, default="dataset/videomme_long/compressed", help="Output cache directory")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers (default: 32)")
    parser.add_argument("--max-mb", type=int, default=85, help="Max video size in MB (default: 85)")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only encoding")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    max_bytes = args.max_mb * 1024 * 1024

    if not video_dir.exists():
        print(f"Error: {video_dir} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = (not args.no_gpu) and _detect_nvenc()
    encoder = "h264_nvenc (GPU)" if use_gpu else "libx264 (CPU)"

    exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in exts])

    if not videos:
        print(f"No videos found in {video_dir}")
        sys.exit(0)

    print(f"Found {len(videos)} videos in {video_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Encoder: {encoder}, Workers: {args.workers}, Max size: {args.max_mb}MB\n", flush=True)

    stats = {"cached": 0, "symlink": 0, "compressed": 0, "error": 0}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, str(v), str(output_dir), max_bytes, use_gpu): v
            for v in videos
        }

        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            elapsed = r.get("time", 0)

            if r["status"] == "cached":
                print(f"  [{i}/{len(videos)}] CACHED  {r['name']} ({r['final_mb']:.0f}MB)", flush=True)
            elif r["status"] == "symlink":
                print(f"  [{i}/{len(videos)}] OK      {r['name']} ({r['original_mb']:.0f}MB)", flush=True)
            elif r["status"] == "compressed":
                print(f"  [{i}/{len(videos)}] COMPRESSED {r['name']} ({r['original_mb']:.0f}MB → {r['final_mb']:.0f}MB) [{elapsed:.1f}s]", flush=True)
            else:
                print(f"  [{i}/{len(videos)}] ERROR   {r['name']}: {r.get('error', '?')}", flush=True)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s", flush=True)
    print(f"  Cached: {stats['cached']}, Symlinked: {stats['symlink']}, "
          f"Compressed: {stats['compressed']}, Errors: {stats['error']}", flush=True)


if __name__ == "__main__":
    main()
