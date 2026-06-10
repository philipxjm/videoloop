import ast
import json
import os
import re
import time
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_TRAJ_RE = re.compile(
    r"TrajectoryStep\(step_type='([^']*)',\s*content=(.*),\s*timestamp=([\d.]+)\)"
)
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from video_agent.token_tracker import aggregate_token_usage

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"


def _parse_trajectory_step(step):
    """Parse a TrajectoryStep string repr into a dict the frontend expects.

    Worker results store trajectory as a list of strings like:
        "TrajectoryStep(step_type='tool_use', content={...}, timestamp=1234)"
    The frontend needs {step_type, content, timestamp} dicts.
    """
    if isinstance(step, dict):
        return step  # already parsed
    if not isinstance(step, str):
        return {"step_type": "unknown", "content": str(step)}
    m = _TRAJ_RE.match(step)
    if not m:
        return {"step_type": "unknown", "content": step}
    step_type = m.group(1)
    content_str = m.group(2)
    timestamp = float(m.group(3))
    try:
        content = ast.literal_eval(content_str)
    except Exception:
        content = content_str
    return {"step_type": step_type, "content": content, "timestamp": timestamp}


def _normalize_trajectory(question: dict) -> dict:
    """Ensure trajectory steps are dicts, not string reprs."""
    traj = question.get("trajectory")
    if traj and isinstance(traj, list) and traj and isinstance(traj[0], str):
        question["trajectory"] = [_parse_trajectory_step(s) for s in traj]
    return question
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_LOG = LOG_DIR / "eval_live.log"
ACTIVE_RUN_FILE = LOG_DIR / "active_run.json"
DEFAULT_PORT = 8080


def _active_run() -> dict | None:
    """Read logs/active_run.json (written by agent_cli at run start).

    Returns the parsed dict, or None if the file is missing/invalid.
    The pointer file is the source of truth for which run the dashboard
    serves; it survives restarts of the uvicorn server because each
    request re-reads it.
    """
    if not ACTIVE_RUN_FILE.exists():
        return None
    try:
        return json.loads(ACTIVE_RUN_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _active_run_dir() -> Path:
    """Return the active run's output_dir, falling back to LOG_DIR root."""
    info = _active_run()
    if info and info.get("output_dir"):
        p = Path(info["output_dir"])
        if p.exists():
            return p
    return LOG_DIR


def _dashboard_state_path() -> Path:
    return _active_run_dir() / "dashboard_questions.json"


def _live_question_path() -> Path:
    return _active_run_dir() / "live_question.json"

PROGRESS_RE = re.compile(
    r"Progress:\s+(\d+)/(\d+)\s+\|\s+Accuracy:\s+([\d.]+)%\s+\|\s+Rate:\s+([\d.]+)\s*s/q"
)
LEGACY_PROGRESS_RE = re.compile(
    r"Progress:\s+(\d+)/(\d+)\s+\|\s+Accuracy:\s+([\d.]+)%\s+\|\s+Rate:\s+([\d.]+)\s*q/min"
)
VIDEO_RE = re.compile(r"VIDEO\s+(\d+)/(\d+):\s+(.+)")
QUESTION_RE = re.compile(r"Question\s+(\d+)/(\d+):\s+(.+)")
TOTAL_RE = re.compile(r"Total questions:\s+(\d+)")
SUMMARY_RATE_RE = re.compile(r"Rate:\s+([\d.]+)\s+seconds/question")

app = FastAPI(title="VideoMME Live Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory cache for dashboard_questions.json — only re-parsed when the file changes
_qs_cache: dict = {"data": None, "mtime": 0.0}
# Cooldown: don't re-read worker_results more often than every 10 seconds
_worker_results_cooldown: float = 0.0


def resolve_log_path() -> Path:
    """Determine which log file to read from.

    Resolution order:
      1. EVAL_LOG_PATH env var (set by agent_cli when it spawns its own dashboard)
      2. active_run.json's log_file (written by every parallel-eval startup)
      3. logs/eval_live.log (legacy symlink)

    The previous glob fallback over ``eval_*.log`` was removed because it
    silently resurrected ancient runs whenever the live pointers were absent.
    """
    env_path = os.getenv("EVAL_LOG_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    info = _active_run()
    if info and info.get("log_file"):
        candidate = Path(info["log_file"])
        if candidate.exists():
            return candidate

    if DEFAULT_LOG.exists():
        return DEFAULT_LOG

    raise FileNotFoundError("No evaluation log file found")


def tail_lines(path: Path, limit: int = 200) -> List[str]:
    """Return the last `limit` lines from a log file efficiently."""
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    dq: Deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


def parse_status(lines: List[str]):
    """Parse the latest progress, accuracy, and rate metrics from log lines."""
    progress_line = None
    video_line = None
    question_line = None
    summary_rate_line = None
    total_questions = None

    for line in reversed(lines):
        clean_line = ANSI_RE.sub("", line)
        if progress_line is None and "Progress:" in line:
            progress_line = clean_line
        if video_line is None and "VIDEO" in line:
            video_line = clean_line
        if question_line is None and "Question" in line:
            question_line = clean_line
        if summary_rate_line is None and "seconds/question" in line:
            summary_rate_line = clean_line
        if total_questions is None and "Total questions:" in line:
            total_questions = clean_line
        if progress_line and video_line and question_line and summary_rate_line and total_questions:
            break

    progress = None
    accuracy = None
    rate_s_per_q = None
    answered = None
    total = None

    if progress_line:
        match = PROGRESS_RE.search(progress_line)
        if match:
            answered = int(match.group(1))
            total = int(match.group(2))
            accuracy = float(match.group(3)) / 100.0
            rate_s_per_q = float(match.group(4))
        else:
            legacy_match = LEGACY_PROGRESS_RE.search(progress_line)
            if legacy_match:
                answered = int(legacy_match.group(1))
                total = int(legacy_match.group(2))
                accuracy = float(legacy_match.group(3)) / 100.0
                # Legacy log recorded rate as q/min but computed q/sec. Convert to s/q conservatively.
                legacy_rate = float(legacy_match.group(4))
                if legacy_rate > 0:
                    rate_s_per_q = 1.0 / legacy_rate

    if total_questions is None and total is not None:
        total_questions = total
    elif isinstance(total_questions, str):
        total_match = TOTAL_RE.search(total_questions)
        if total_match:
            total_questions = int(total_match.group(1))

    if summary_rate_line and rate_s_per_q is None:
        summary_match = SUMMARY_RATE_RE.search(summary_rate_line)
        if summary_match:
            rate_s_per_q = float(summary_match.group(1))

    video = None
    if video_line:
        vm = VIDEO_RE.search(video_line)
        if vm:
            video = {
                "index": int(vm.group(1)),
                "total": int(vm.group(2)),
                "label": vm.group(3).strip(),
            }

    question = None
    if question_line:
        qm = QUESTION_RE.search(question_line)
        if qm:
            question = {
                "index": int(qm.group(1)),
                "total": int(qm.group(2)),
                "label": qm.group(3).strip(),
            }

    status = {
        "progress": {
            "answered": answered,
            "total": total,
            "accuracy": accuracy,
            "rate_seconds_per_question": rate_s_per_q,
        },
        "video": video,
        "question": question,
        "log_timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if total_questions:
        status["progress"]["total_questions"] = (
            total_questions if isinstance(total_questions, int) else None
        )

    if answered is not None and accuracy is not None:
        estimated_correct = int(round(answered * accuracy))
        status["progress"]["estimated_correct"] = estimated_correct

    return status


def _find_final_results() -> Path | None:
    """Find the most recent parallel_eval_*.json results file in the active run dir."""
    run_dir = _active_run_dir()
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.glob("parallel_eval_*.json"))
    return candidates[-1] if candidates else None


def _load_from_results_file(path: Path) -> dict:
    """Load question state from a final results file (parallel_eval_*.json).

    Adapts the results file format (summary + results) to the dashboard
    format (meta + questions) so the same frontend code works for both
    live and post-run views.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    summary = raw.get("summary", {})
    results = raw.get("results", [])

    # Build meta from summary
    meta = {
        "run_id": path.stem.replace("parallel_eval_", ""),
        "model": summary.get("model"),
        "answered": summary.get("answered", 0),
        "total_questions": summary.get("total_questions", 0),
        "correct": summary.get("correct", 0),
        "failed": summary.get("failed", 0),
        "accuracy": summary.get("accuracy", 0),
        "rate_s_per_q": summary.get("rate_seconds_per_question"),
        "elapsed_seconds": summary.get("elapsed_seconds", 0),
        "num_workers": summary.get("num_workers", 0),
        "status": "completed",
    }

    # Aggregate token usage
    agg, _ = aggregate_token_usage(results)
    meta["token_usage"] = agg

    return {"meta": meta, "questions": results}


def _parse_dashboard_meta() -> dict | None:
    """Parse just the meta block from dashboard_questions.json.

    The file starts with {"meta": {...}, "questions": [...]} — the meta object
    is always fully written before questions start, so it's valid JSON even
    when the file is truncated mid-questions-array.
    """
    state_path = _dashboard_state_path()
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as f:
            # Read enough to capture the full meta block (typically <2KB)
            chunk = f.read(8192)
        # Find the meta object boundaries
        meta_start = chunk.find('"meta"')
        if meta_start == -1:
            return None
        # Find the opening brace of the meta value
        brace_start = chunk.find('{', meta_start + 6)
        if brace_start == -1:
            return None
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(chunk)):
            if chunk[i] == '{':
                depth += 1
            elif chunk[i] == '}':
                depth -= 1
                if depth == 0:
                    meta_json = chunk[brace_start:i + 1]
                    return json.loads(meta_json)
        return None
    except Exception:
        return None


def _load_from_worker_results() -> dict | None:
    """Aggregate completed questions from individual worker_results_*.json files.

    These files are written per-worker during parallel evaluation and serve as
    a reliable fallback when dashboard_questions.json is mid-write or corrupt.
    """
    worker_files = sorted(_active_run_dir().glob("worker_results_*.json"))
    if not worker_files:
        return None

    questions = []
    latest_mtime = 0.0
    meta_info = {}
    for wf in worker_files:
        try:
            latest_mtime = max(latest_mtime, wf.stat().st_mtime)
            with wf.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Worker result files can be a single result dict or a list
            if isinstance(data, dict):
                # Could be a single question result or a wrapper
                if "results" in data:
                    questions.extend(data["results"])
                    if not meta_info and "summary" in data:
                        meta_info = data["summary"]
                elif "question_id" in data:
                    questions.append(data)
            elif isinstance(data, list):
                questions.extend(data)
        except (json.JSONDecodeError, Exception):
            continue

    if not questions:
        return None

    # Build aggregate meta
    correct = sum(1 for q in questions if q.get("correct"))
    total = len(questions)

    # Aggregate token usage across all questions
    agg_tokens, agg_by_agent = aggregate_token_usage(questions)

    meta = {
        "answered": total,
        "total_questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "status": "in_progress",
        "token_usage": agg_tokens,
        "token_usage_by_agent": agg_by_agent if agg_by_agent else None,
        **meta_info,
    }

    return {"meta": meta, "questions": questions, "_mtime": latest_mtime}


def load_question_state():
    global _qs_cache

    # Prefer live dashboard state if it exists (run in progress)
    state_path = _dashboard_state_path()
    if state_path.exists():
        try:
            mtime = state_path.stat().st_mtime
            if _qs_cache["data"] is not None and mtime <= _qs_cache["mtime"]:
                return _qs_cache["data"]
            with state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                pass  # fall through to results file
            else:
                data.setdefault("meta", {})
                data.setdefault("questions", [])
                _qs_cache = {"data": data, "mtime": mtime}
                return data
        except json.JSONDecodeError:
            global _worker_results_cooldown
            # dashboard_questions.json is corrupt/mid-write — try worker results.
            # Use a cooldown to avoid re-reading 64 files on every request.
            now = time.time()
            if _qs_cache["data"] is not None and now - _worker_results_cooldown < 10:
                return _qs_cache["data"]
            worker_data = _load_from_worker_results()
            if worker_data:
                worker_data.pop("_mtime", None)
                # The meta from dashboard_questions.json header is valid even
                # when the full file is corrupt (truncated mid-questions-array).
                # Parse it to get the real total_questions count.
                dash_meta = _parse_dashboard_meta()
                if dash_meta:
                    worker_data["meta"] = {**worker_data["meta"], **dash_meta}
                _qs_cache = {"data": worker_data, "mtime": now}
                _worker_results_cooldown = now
                return worker_data
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load dashboard state: {exc}")

    # Fall back to final results file (post-run)
    results_path = _find_final_results()
    if results_path:
        try:
            mtime = results_path.stat().st_mtime
            if _qs_cache["data"] is not None and mtime <= _qs_cache["mtime"]:
                return _qs_cache["data"]
            data = _load_from_results_file(results_path)
            _qs_cache = {"data": data, "mtime": mtime}
            return data
        except (json.JSONDecodeError, Exception):
            pass

    # Last resort: try worker results even without dashboard_questions.json
    worker_data = _load_from_worker_results()
    if worker_data:
        mtime = worker_data.pop("_mtime", time.time())
        _qs_cache = {"data": worker_data, "mtime": mtime}
        return worker_data

    return {"meta": {}, "questions": []}


@app.get("/api/config")
def api_config():
    """Serve pricing and other config to the frontend."""
    try:
        import yaml
        config_path = BASE_DIR / "configs" / "config.yaml"
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        pricing = raw.get("pricing", {})
        vlm = pricing.pop("vlm", {})
        summarizer = pricing.pop("summarizer", {})
        return {
            "pricing": {
                "input_per_million": pricing.get("input_per_million", 0),
                "output_per_million": pricing.get("output_per_million", 0),
                "currency": pricing.get("currency", "USD"),
                "vlm": {
                    "input_per_million": vlm.get("input_per_million", 0),
                    "output_per_million": vlm.get("output_per_million", 0),
                },
                "summarizer": {
                    "input_per_million": summarizer.get("input_per_million", 0),
                    "output_per_million": summarizer.get("output_per_million", 0),
                },
            }
        }
    except Exception:
        return {
            "pricing": {
                "input_per_million": 0,
                "output_per_million": 0,
                "currency": "USD",
                "vlm": {"input_per_million": 0, "output_per_million": 0},
                "summarizer": {"input_per_million": 0, "output_per_million": 0},
            }
        }


@app.get("/api/status")
def api_status():
    try:
        log_path = resolve_log_path()
        lines = tail_lines(log_path, limit=400)
        status = parse_status(lines)
        status["log_file"] = str(log_path)
        return status
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/")
def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard assets missing")
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/questions")
def api_questions():
    data = load_question_state()
    return {
        "meta": data.get("meta", {}),
        # trajectory and reasoning are intentionally excluded here — they can be
        # hundreds of KB per question and are only needed in the detail view.
        # The client fetches them on demand via GET /api/questions/{id}.
        "questions": [
            {
                "question_id": q.get("question_id"),
                "video_id": q.get("video_id"),
                "question": q.get("question"),
                "options": q.get("options"),
                "expected": q.get("expected"),
                "predicted": q.get("predicted"),
                "correct": q.get("correct"),
                "answered_index": q.get("answered_index"),
                "token_usage": q.get("token_usage"),
                "error": q.get("error"),
                "worker_id": q.get("worker_id"),
                "elapsed_seconds": q.get("elapsed_seconds"),
            }
            for q in data.get("questions", [])
        ],
    }


@app.get("/api/questions/{question_id}")
def api_question_detail(question_id: str):
    data = load_question_state()
    for q in data.get("questions", []):
        if str(q.get("question_id")) == str(question_id):
            return _normalize_trajectory(q)
    raise HTTPException(status_code=404, detail="Question not found")


@app.get("/api/live")
def api_live_question():
    """Get the currently active question being worked on (single worker mode)."""
    live_path = _live_question_path()
    if not live_path.exists():
        return {"active": False, "question": None}
    try:
        with live_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"active": True, "question": data}
    except (json.JSONDecodeError, Exception):
        return {"active": False, "question": None}


def _read_live_workers() -> list:
    """Read all live worker files and return full worker dicts."""
    workers = []
    run_dir = _active_run_dir()
    for f in run_dir.glob("live_question_worker_*.json"):
        try:
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["worker_file"] = f.name
                workers.append(data)
        except (json.JSONDecodeError, Exception):
            pass

    live_path = _live_question_path()
    if live_path.exists() and not workers:
        try:
            with live_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                data["worker_id"] = 0
                workers.append(data)
        except (json.JSONDecodeError, Exception):
            pass

    workers.sort(key=lambda w: w.get("worker_id", 0))
    return workers


_WORKER_LIGHT_KEYS = (
    "worker_id", "question_id", "video_id", "question", "options",
    "expected", "predicted", "correct", "token_usage", "error",
    "elapsed_seconds", "worker_file", "status", "num_questions",
)


@app.get("/api/live/workers")
def api_live_workers():
    """Get all currently active workers (lightweight — no trajectory/reasoning)."""
    workers = _read_live_workers()
    light = []
    for w in workers:
        lw = {k: w.get(k) for k in _WORKER_LIGHT_KEYS}
        # Include trajectory step count so the UI can show progress
        lw["trajectory_steps"] = len(w.get("trajectory") or [])
        light.append(lw)
    return {
        "active": len(light) > 0,
        "num_workers": len(light),
        "workers": light,
    }


@app.get("/api/live/workers/{worker_id}")
def api_live_worker_detail(worker_id: int):
    """Get full detail for a single live worker (including trajectory)."""
    workers = _read_live_workers()
    for w in workers:
        if w.get("worker_id") == worker_id:
            return _normalize_trajectory(w)
    raise HTTPException(status_code=404, detail="Worker not found")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="VideoMME Live Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to serve on (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

