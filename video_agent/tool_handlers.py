"""
Tool handler registry for the Video Understanding Agent.

Each handler takes (agent, input_data) -> str and is registered
by tool name. The agent instance provides access to docker, vlm,
memory, etc.
"""

import os
import json
import base64
import glob
import re
import shlex
import subprocess
import tempfile

from .logging_utils import get_logger

logger = get_logger(__name__)


def _shell_quote(cmd: str) -> str:
    """Quote a command string for use as an argument to bash -c."""
    return shlex.quote(cmd)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_HANDLERS = {}


def _register(name):
    """Decorator to register a tool handler by name."""
    def decorator(fn):
        _HANDLERS[name] = fn
        return fn
    return decorator


def dispatch(agent, name: str, input_data: dict) -> str:
    """Look up and call the handler for *name*. Returns result string."""
    # Enforce include_tools whitelist at dispatch level (catches text-parsed calls)
    include = getattr(agent, '_include_tools', None)
    if include and name not in include and name not in ("pin_memory", "complete_todo"):
        return f"Error: Tool '{name}' is not available. Use one of: {', '.join(include)}"
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Error: Unknown tool '{name}'"
    return handler(agent, input_data)


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------


def _extract_clip_frames(clip_path: str, fps: float = 1.0) -> list:
    """Extract all frames from a clip at the given fps, return as base64 data URLs.

    Matches the same 1fps/480p encoding the clip was created with, so the main
    agent sees every frame the VLM subagent would have seen.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="clip_frames_") as tmpdir:
            subprocess.run(
                ["ffmpeg", "-y", "-i", clip_path,
                 "-vf", f"fps={fps},scale=-2:480",
                 "-q:v", "5",
                 os.path.join(tmpdir, "frame_%04d.jpg")],
                capture_output=True, timeout=60,
            )
            frame_files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
            urls = []
            for fpath in frame_files:
                with open(fpath, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                    urls.append(f"data:image/jpeg;base64,{b64}")
            logger.info(f"Extracted {len(urls)} frames from clip at {fps}fps")
            return urls
    except Exception as e:
        logger.warning(f"Clip frame extraction failed: {e}")
        return []


_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _frames_to_base64(local_frame_paths: list, max_frames: int = 8) -> list:
    """Convert local frame image files to base64 data URLs."""
    urls = []
    for fpath in local_frame_paths[:max_frames]:
        try:
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in _VALID_IMAGE_EXTS:
                logger.warning(f"Skipping non-image file: {fpath}")
                continue
            file_size = os.path.getsize(fpath)
            if file_size > 5 * 1024 * 1024:  # 5MB guard
                logger.warning(f"Skipping oversized file ({file_size/1024/1024:.1f}MB): {fpath}")
                continue
            with open(fpath, "rb") as fh:
                data = fh.read()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
                ext.lstrip("."), "image/jpeg"
            )
            b64 = base64.b64encode(data).decode()
            urls.append(f"data:{mime};base64,{b64}")
        except Exception as e:
            logger.warning(f"Failed to read frame {fpath}: {e}")
    return urls


def _make_multimodal_result(text: str, image_urls: list) -> list:
    """Build OpenAI-compatible multimodal content list from text + image URLs."""
    parts = [{"type": "text", "text": text}]
    for url in image_urls:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts



# ---------------------------------------------------------------------------
# Video analysis tools
# ---------------------------------------------------------------------------


@_register("transcribe_audio")
def _transcribe_audio(agent, input_data):
    result = agent._docker.transcribe_audio(
        video_path=input_data["video_path"],
        model_size=input_data.get("model_size", "base"),
    )
    # Save transcript inside the container so the agent can grep/search it
    transcript_text = result.get("transcript", "")
    if transcript_text and agent._docker:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(transcript_text)
                tmp_path = f.name
            agent._docker.copy_to_container(tmp_path, "/outputs/transcript.txt")
            os.unlink(tmp_path)
            logger.info("Wrote transcript to /outputs/transcript.txt in container")
        except Exception as e:
            logger.warning(f"Failed to write transcript to container: {e}")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Visual analysis tools
# ---------------------------------------------------------------------------

MAX_FRAMES_PER_BATCH = 64  # Max frames sent to the model in a single visual-analysis call


@_register("analyze_frames")
def _analyze_frames(agent, input_data):
    frame_paths = input_data.get("frame_paths") or input_data.get("image_paths", [])
    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]
    # Filter out non-image files (e.g., .mp4 video files passed by mistake)
    skipped = [fp for fp in frame_paths if os.path.splitext(fp)[1].lower() not in _VALID_IMAGE_EXTS]
    if skipped:
        logger.warning(f"analyze_frames: skipping non-image files: {skipped}")
    image_frame_paths = [fp for fp in frame_paths if os.path.splitext(fp)[1].lower() in _VALID_IMAGE_EXTS]
    if not image_frame_paths:
        return (
            f"Error: No valid image files provided. Got: {frame_paths}. "
            "You must extract frames first using execute_bash with ffmpeg, e.g.: "
            "ffmpeg -ss 120 -i /videos/video.mp4 -frames:v 1 /outputs/frame_t120.jpg"
        )
    local_frames = [f for f in (agent._copy_frame_to_host(fp) for fp in image_frame_paths) if f]
    if not local_frames:
        return "Error: Could not access any frames"

    if getattr(agent, '_multimodal_agent', False):
        # Native multimodal: return images directly, agent analyzes with full memory context
        # Split into batches of MAX_FRAMES_PER_BATCH to stay under 16-image API limit
        image_urls = _frames_to_base64(local_frames, max_frames=len(local_frames))
        if image_urls:
            # Cache base64 by Docker path for render_context
            if not hasattr(agent, '_frame_b64_cache'):
                agent._frame_b64_cache = {}
            for docker_path, url in zip(frame_paths, image_urls):
                if ';base64,' in url:
                    agent._frame_b64_cache[docker_path] = url.split(';base64,', 1)[1]

            question = input_data.get("question", "Describe what you observe.")

            # Split into batches
            if len(image_urls) <= MAX_FRAMES_PER_BATCH:
                # Single batch — return directly
                return _make_multimodal_result(
                    f"[{len(image_urls)} frames attached. {question}]",
                    image_urls,
                )
            else:
                # Multiple batches — store remaining on agent for sequential processing
                batches = []
                for i in range(0, len(image_urls), MAX_FRAMES_PER_BATCH):
                    batch_urls = image_urls[i:i + MAX_FRAMES_PER_BATCH]
                    batch_paths = frame_paths[i:i + MAX_FRAMES_PER_BATCH]
                    batches.append((batch_urls, batch_paths))

                # Store remaining batches for _process_response to handle
                if not hasattr(agent, '_pending_frame_batches'):
                    agent._pending_frame_batches = []
                for batch_urls, batch_paths in batches[1:]:
                    agent._pending_frame_batches.append({
                        'image_urls': batch_urls,
                        'frame_paths': batch_paths,
                        'question': question,
                        'original_input': input_data,
                    })

                logger.info(
                    f"[FrameBatch] Split {len(image_urls)} frames into {len(batches)} batches "
                    f"({MAX_FRAMES_PER_BATCH} per batch, {len(agent._pending_frame_batches)} pending)"
                )

                # Return first batch
                first_urls = batches[0][0]
                return _make_multimodal_result(
                    f"[Batch 1/{len(batches)}: {len(first_urls)} frames attached. {question}]",
                    first_urls,
                )

    # VLM subagent fallback
    question = input_data["question"]
    response = agent.vlm.ask(local_frames, question)
    return response


@_register("analyze_clip")
def _analyze_clip(agent, input_data):
    """Extract a video clip and send it natively to the VLM."""
    start_time = float(input_data.get("start_time", 0))
    end_time = float(input_data.get("end_time", 0))
    question = input_data.get("question", "Describe what happens in this clip.")

    duration = end_time - start_time
    if duration <= 0:
        return "Error: end_time must be greater than start_time"
    if duration > 600:
        return "Error: clip duration exceeds 10 minutes; use shorter clips"

    # Find the video path in the container
    video_path = getattr(agent, '_container_video_path', None)
    if not video_path:
        # Try to find the video — in group mode it's typically /videos/<name>
        exit_code, ls = agent._docker.execute("ls /videos/*.mp4 2>/dev/null | head -1", timeout=5)
        if exit_code == 0 and ls.strip():
            video_path = ls.strip()
        else:
            return "Error: video path not set on agent; cannot extract clip"

    clip_container_path = f"/tmp/vlm_clip_{int(start_time)}_{int(end_time)}.mp4"
    ffmpeg_cmd = (
        f"ffmpeg -y -ss {start_time:.3f} -i {video_path} "
        f"-t {duration:.3f} "
        f"-vf 'scale=-2:480' "
        f"-b:v 500k -c:a aac -b:a 64k -ac 1 "
        f"{clip_container_path} 2>/dev/null && "
        f"stat --format='%s' {clip_container_path}"
    )
    ffmpeg_timeout = max(60, int(duration * 2))
    exit_code, output = agent._docker.execute(ffmpeg_cmd, timeout=ffmpeg_timeout)
    if exit_code != 0:
        return f"Error extracting clip: {output[:200]}"

    # Copy clip out of container
    if agent._temp_dir is None:
        agent._temp_dir = tempfile.mkdtemp(prefix="video_agent_")
    local_clip_path = os.path.join(agent._temp_dir, os.path.basename(clip_container_path))
    if not agent._docker.copy_from_container(clip_container_path, local_clip_path):
        return "Error: Could not copy clip from container"
    if not os.path.exists(local_clip_path):
        return "Error: Clip file not found after copy"

    clip_size_mb = os.path.getsize(local_clip_path) / (1024 * 1024)
    logger.info(f"Clip extracted: {start_time:.0f}s-{end_time:.0f}s, {clip_size_mb:.1f}MB")

    # Check size — Gemini accepts up to ~95MB inline_data
    if clip_size_mb > 80:
        try:
            os.unlink(local_clip_path)
        except OSError:
            pass
        return f"Error: Clip too large ({clip_size_mb:.0f}MB). Try a shorter duration (<60s)."

    # Multimodal agent path: return video inline for agent to analyze directly
    if getattr(agent, '_multimodal_agent', False):
        try:
            with open(local_clip_path, 'rb') as f:
                clip_b64 = base64.b64encode(f.read()).decode()
            os.unlink(local_clip_path)
        except OSError:
            return f"Error: Could not read clip file"
        return [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{clip_b64}"}},
            {"type": "text", "text": f"[Clip {start_time:.0f}s-{end_time:.0f}s. Question: {question}]"},
        ]

    # VLM subagent fallback
    if hasattr(agent, 'vlm') and hasattr(agent.vlm, 'ask_about_clip'):
        response = agent.vlm.ask_about_clip(local_clip_path, question)
        try:
            os.unlink(local_clip_path)
        except OSError:
            pass
        return response

    return "Error: No method available to analyze video clip (need multimodal agent or VLM with ask_about_clip)"


# ---------------------------------------------------------------------------
# File system tools
# ---------------------------------------------------------------------------

@_register("execute_bash")
def _execute_bash(agent, input_data):
    # Snapshot existing image files before execution
    _, pre_files = agent._docker.execute(
        "find /outputs /tmp -maxdepth 2 -name '*.jpg' -o -name '*.png' 2>/dev/null", timeout=5
    )
    pre_set = set(pre_files.strip().split('\n')) if pre_files.strip() else set()

    # Wrap command with timeout to prevent ffmpeg/long processes from hanging
    cmd = input_data["command"]
    exit_code, output = agent._docker.execute(f"timeout 120 bash -c {_shell_quote(cmd)}")

    # Check for newly created image files and cache them
    _, post_files = agent._docker.execute(
        "find /outputs /tmp -maxdepth 2 -name '*.jpg' -o -name '*.png' 2>/dev/null", timeout=5
    )
    post_set = set(post_files.strip().split('\n')) if post_files.strip() else set()
    new_images = post_set - pre_set
    if new_images:
        if not hasattr(agent, '_frame_b64_cache'):
            agent._frame_b64_cache = {}
        for img_path in new_images:
            if not img_path.strip():
                continue
            ec, b64 = agent._docker.execute(f"base64 -w0 {img_path} 2>/dev/null", timeout=10)
            if ec == 0 and b64 and not b64.startswith("[STDERR]"):
                agent._frame_b64_cache[img_path.strip()] = b64.strip()
                logger.info(f"[FrameEmbed] Cached new image: {img_path.strip()}")

    # Clean up verbose tool output — strip ffmpeg/build noise, keep useful info
    output = _clean_bash_output(output, new_images)
    return output


def _clean_bash_output(output: str, new_images: set = None) -> str:
    """Strip noisy stderr from bash output, keep useful content."""
    if not output:
        return output

    lines = output.split('\n')
    cleaned = []
    skip_ffmpeg_header = False

    for line in lines:
        stripped = line.strip()

        # Skip ffmpeg build/config noise
        if stripped.startswith('[STDERR]'):
            # Keep the STDERR marker but filter the content after it
            rest = stripped[8:].strip()
            # Skip ffmpeg version/build info
            if any(rest.startswith(p) for p in (
                'ffmpeg version', 'built with', 'configuration:', 'lib',
                'Copyright', 'the FFmpeg developers',
            )):
                skip_ffmpeg_header = True
                continue
            # Skip stream/codec info
            if any(p in rest for p in (
                'Stream #', 'Stream mapping', 'Press [q]',
                'Output #', 'Input #', 'Duration:', 'encoder',
                'Metadata:', 'handler_name', 'compatible_brands',
                'major_brand', 'minor_version', 'creation_time',
                'bitrate:', 'start:', 'SAR', 'DAR',
            )):
                continue
            # Skip frame progress lines
            if rest.startswith('frame=') or rest.startswith('size='):
                continue
            # Keep actual errors
            if rest:
                skip_ffmpeg_header = False
                cleaned.append(line)
            continue

        skip_ffmpeg_header = False
        cleaned.append(line)

    result = '\n'.join(cleaned).strip()

    # Append summary of new files created
    if new_images:
        valid = sorted(f for f in new_images if f.strip())
        if valid:
            result += f"\n[Created {len(valid)} image(s): {', '.join(valid[:5])}]"
            if len(valid) > 5:
                result += f" ...and {len(valid) - 5} more"

    return result


@_register("list_files")
def _list_files(agent, input_data):
    exit_code, output = agent._docker.execute(f"ls -la {input_data['directory']}")
    return output


@_register("read_file")
def _read_file(agent, input_data):
    exit_code, output = agent._docker.execute(f"cat {input_data['file_path']}")
    return output


@_register("create_file")
def _create_file(agent, input_data):
    path = input_data.get("path", "")
    content = input_data.get("content", "")
    if not path or not content:
        return "Error: Both 'path' and 'content' are required."
    content_b64 = base64.b64encode(content.encode()).decode()
    cmd = f"echo '{content_b64}' | base64 -d > {path}"
    exit_code, output = agent._docker.execute(cmd)
    if exit_code != 0:
        return f"Error creating file: {output}"
    return f'File created: {path}. Run with: execute_bash(command="python {path}")'


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

@_register("pin_memory")
def _pin_memory(agent, input_data):
    if not hasattr(agent, '_current_memory') or agent._current_memory is None:
        return "Error: Memory system not enabled."
    memory = agent._current_memory
    if hasattr(memory, 'pin_entry'):
        return memory.pin_entry(
            content=input_data.get("content"),
            entry_id=input_data.get("entry_id"),
            start_t=input_data.get("start_time"),
            end_t=input_data.get("end_time"),
        )
    return "Error: pin_memory requires orchestrated or file-backed memory."


# ---------------------------------------------------------------------------
# Visual planner todo tools
# ---------------------------------------------------------------------------

@_register("complete_todo")
def _complete_todo(agent, input_data):
    memory = getattr(agent, '_current_memory', None)
    if memory is None or not getattr(memory, '_todo_list', None):
        return "No todo list in memory."
    try:
        todo_id = int(input_data.get("id", -1))
    except (ValueError, TypeError):
        return "Invalid todo ID."
    finding = str(input_data.get("finding", "")).strip()
    if not finding:
        return "A 'finding' is required to complete a todo."

    for item in memory._todo_list:
        if item.get("id") == todo_id:
            if item.get("done"):
                return f"Todo {todo_id} is already marked complete."
            # Verify with the planner
            approved, feedback = agent._verify_todo(item, finding)
            if approved:
                item["done"] = True
                item["finding"] = finding
                item.pop("last_attempt", None)
                item.pop("feedback", None)
                remaining = sum(1 for t in memory._todo_list if not t.get("done"))
                return f"Todo {todo_id} verified and closed. {remaining} remaining."
            else:
                item["last_attempt"] = finding
                item["feedback"] = feedback
                return f"Todo {todo_id} not yet complete: {feedback}"
    return f"Todo ID {todo_id} not found."


# ---------------------------------------------------------------------------
# Answer submission
# ---------------------------------------------------------------------------

@_register("submit_answer")
def _submit_answer(agent, input_data):
    return json.dumps({
        "status": "answer_submitted",
        "answer": input_data["answer"],
        "reasoning": input_data.get("reasoning", ""),
    })
