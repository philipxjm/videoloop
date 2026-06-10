#!/usr/bin/env python3
"""Unified CLI for the video understanding agent."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Load .env before any other imports that might read env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from video_agent import VideoUnderstandingAgent
from video_agent.config import get_config

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "dataset/videomme_long/videomme_long_full.json"
DEFAULT_VIDEO_DIR = Path("dataset/videomme_long/videos")
DEFAULT_OUTPUT = BASE_DIR / "logs"

# Global to track dashboard process for cleanup
_dashboard_process: Optional[subprocess.Popen] = None


def _start_dashboard(log_file: Path, port: int = 8080) -> Optional[subprocess.Popen]:
    """Start the dashboard server in the background."""
    global _dashboard_process
    
    dashboard_script = BASE_DIR / "dashboard" / "server.py"
    if not dashboard_script.exists():
        return None
    
    # Set environment variable for the log file
    env = os.environ.copy()
    env["EVAL_LOG_PATH"] = str(log_file)
    
    try:
        # Start uvicorn in background
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "dashboard.server:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--log-level", "warning",
            ],
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _dashboard_process = proc
        return proc
    except Exception:
        return None


def _stop_dashboard():
    """Stop the dashboard server if running."""
    global _dashboard_process
    if _dashboard_process is not None:
        try:
            _dashboard_process.terminate()
            _dashboard_process.wait(timeout=5)
        except Exception:
            try:
                _dashboard_process.kill()
            except Exception:
                pass
        _dashboard_process = None


def _register_active_run(output_dir: Path, log_file: Path, run_id: str, dataset: Optional[str] = None):
    """Write logs/active_run.json so the persistent dashboard finds the live run.

    The dashboard reads this pointer on each request, so any out-of-band uvicorn
    instance (started before this run) still resolves to the correct output_dir
    and log file without needing EVAL_LOG_PATH env or hand-curated symlinks.
    """
    log_root = DEFAULT_OUTPUT
    log_root.mkdir(parents=True, exist_ok=True)
    info = {
        "output_dir": str(output_dir.resolve()),
        "log_file": str(log_file.resolve()),
        "run_id": run_id,
        "dataset": dataset,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (log_root / "active_run.json").write_text(json.dumps(info, indent=2))
    except Exception:
        pass


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("agent_cli")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_options(options: Optional[str]) -> Optional[List[str]]:
    if not options:
        return None
    return [opt.strip() for opt in options.split(",") if opt.strip()]


def cmd_single(args):
    agent = VideoUnderstandingAgent(
        model=args.model or None,
        api_base=getattr(args, 'api_base', None),
        api_key=getattr(args, 'api_key', None),
        docker_image=args.docker_image or None,
        verbose=True,
        debug=getattr(args, 'debug', False),
        no_memory=getattr(args, 'no_memory', False),
    )

    option_list = parse_options(args.options)
    result = agent.run(video_path=args.video, question=args.question, options=option_list)

    print("=" * 60)
    if result.success:
        print(f"Answer: {result.answer}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
    else:
        print(f"Failed: {result.error}")
    return 0 if result.success else 1


def cmd_parallel_eval(args):
    """Run parallel evaluation with multiple workers."""
    from video_agent.parallel_eval import ParallelEvaluator
    
    cfg = get_config()
    run_id = f"parallel_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log_file = args.output_dir / f"{run_id}.log"
    logger = setup_logger(log_file)

    # Register this as the active run so the persistent dashboard auto-discovers it.
    _register_active_run(args.output_dir, log_file, args.run_id or run_id, str(args.dataset))

    logger.info("Starting parallel evaluation")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Video dir: {args.video_dir}")
    
    # Start dashboard unless disabled
    dashboard_proc = None
    if not getattr(args, 'no_dashboard', False):
        dashboard_proc = _start_dashboard(log_file, port=args.dashboard_port)
        if dashboard_proc:
            logger.info(f"Dashboard started at http://localhost:{args.dashboard_port}")
            atexit.register(_stop_dashboard)
    
    no_memory = getattr(args, 'no_memory', False)
    if no_memory:
        logger.info("Memory: DISABLED")

    raw_context = getattr(args, 'raw_context', False)
    if raw_context and not no_memory:
        sys.exit("ERROR: --raw-context requires --no-memory (config A only). "
                 "Aborting to prevent contaminating configs B/C.")
    if raw_context:
        logger.info("Layered truncations: DISABLED (raw_context — only max_ctx_tokens applies)")

    no_gate = getattr(args, 'no_early_submit_gate', False)
    early_submit_gate = not no_gate
    if no_gate:
        logger.info("Early-submit gate: DISABLED (transcript / min-iters / min-VLM-calls / open-todo bounces all skipped)")

    max_questions = getattr(args, 'max_questions', None)
    if max_questions:
        logger.info(f"Max questions: {max_questions}")

    # Resolve model overrides: --all-models sets both agent and orchestrator
    agent_model = args.model or cfg.main_agent.model
    orchestrator_model = getattr(args, 'orchestrator_model', None)
    all_models = getattr(args, 'all_models', None)
    if all_models:
        agent_model = all_models
        orchestrator_model = all_models

    logger.info(f"Agent model: {agent_model}")
    if orchestrator_model:
        logger.info(f"Orchestrator model: {orchestrator_model}")
    else:
        logger.info(f"Orchestrator model: {cfg.memory.summarizer_model} (from config)")

    evaluator = ParallelEvaluator(
        num_workers=args.workers,
        model=agent_model,
        api_base=args.api_base or cfg.main_agent.api_base,
        api_key=args.api_key,
        max_iterations=args.max_iterations,
        output_dir=str(args.output_dir),
        no_memory=no_memory,
        orchestrator_model=orchestrator_model,
        max_ctx_tokens=getattr(args, "max_ctx_tokens", None),
        raw_context=raw_context,
        early_submit_gate=early_submit_gate,
    )

    try:
        results = evaluator.run(
            dataset_path=str(args.dataset),
            video_dir=str(args.video_dir),
            dashboard_port=args.dashboard_port,
            run_id=args.run_id,
            max_questions=max_questions,
        )

        summary = results.get("summary", {})
        logger.info(f"Evaluation complete!")
        logger.info(f"Accuracy: {summary.get('correct', 0)}/{summary.get('answered', 0)} = {summary.get('accuracy', 0):.1%}")
        logger.info(f"Time: {summary.get('elapsed_seconds', 0):.1f}s")
        return 0
    finally:
        if dashboard_proc:
            _stop_dashboard()


def build_parser():
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Video understanding agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="Run on a single video/question")
    p_single.add_argument("--video", required=True, help="Path to video file")
    p_single.add_argument("--question", required=True, help="Question to ask")
    p_single.add_argument("--options", help="Comma-separated options (e.g., A,B,C,D)")
    p_single.add_argument("--model", help=f"Model to use (default: {cfg.main_agent.model})")
    p_single.add_argument("--api-base", help="API base URL for main agent")
    p_single.add_argument("--api-key", help="API key for main agent")
    p_single.add_argument("--docker-image", help=f"Docker image (default: {cfg.sandbox.full_image})")
    p_single.add_argument("--debug", action="store_true", help="Enable detailed debug output (API timing, message counts)")
    p_single.add_argument("--no-memory", action="store_true", help="Disable hierarchical memory")
    p_single.set_defaults(func=cmd_single)

    # Parallel evaluation command
    p_parallel = sub.add_parser("parallel-eval", help="Run parallel evaluation with multiple workers")
    p_parallel.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to VideoMME JSON")
    p_parallel.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing video files")
    p_parallel.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory to store logs/results")
    p_parallel.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    p_parallel.add_argument("--model", help=f"Agent model (default: {cfg.main_agent.model})")
    p_parallel.add_argument("--orchestrator-model", help=f"Orchestrator/summarizer model (default: {cfg.memory.summarizer_model})")
    p_parallel.add_argument("--all-models", help="Set both agent and orchestrator to this model")
    p_parallel.add_argument("--api-base", help="API base URL")
    p_parallel.add_argument("--api-key", help="API key")
    p_parallel.add_argument("--max-iterations", type=int, default=50, help="Max iterations per question")
    p_parallel.add_argument("--dashboard-port", type=int, default=8080, help="Port for dashboard server (default: 8080)")
    p_parallel.add_argument("--no-dashboard", action="store_true", help="Disable automatic dashboard startup")
    p_parallel.add_argument("--run-id", help="Custom run ID (default: auto-generated timestamp)")
    p_parallel.add_argument("--no-memory", action="store_true", help="Disable hierarchical memory")
    p_parallel.add_argument("--max-ctx-tokens", type=int, default=None,
                            help="Layer 3 context budget for no_memory mode (default 500_000). "
                                 "Set lower (e.g. 32000) to study tighter budgets.")
    p_parallel.add_argument("--no-early-submit-gate", action="store_true",
                            help="Disable the early-submit gate (min iters / min VLM calls / "
                                 "require transcript / open-todo bounces). Used for ablations "
                                 "that test how the gate shapes trajectory length.")
    p_parallel.add_argument("--raw-context", action="store_true",
                            help="Disable layered truncations (tool-output cap, image strip, "
                                 "overthink trim, transcript pre-trim) so the only ceiling is "
                                 "--max-ctx-tokens. REQUIRES --no-memory. Used for thrashing "
                                 "ablations on config A only — has no effect on configs B/C.")
    p_parallel.add_argument("--max-questions", type=int, help="Cap total questions")
    p_parallel.set_defaults(func=cmd_parallel_eval)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
