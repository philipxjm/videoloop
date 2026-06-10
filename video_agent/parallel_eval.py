"""
Parallel Evaluation Pipeline with Dynamic Job Queue
Workers pull jobs from a shared queue as they complete tasks.
"""

import json
import os
import time
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading

from .token_tracker import aggregate_token_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_int(s: str) -> int:
    """Parse int from string, returning 0 for non-numeric values."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def _normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, units, trailing zeros, and lowercases.
    """
    import re
    s = s.strip()
    # Normalize circled/special Unicode letters to plain ASCII
    circled_map = {
        'ⓐ': 'A', 'ⓑ': 'B', 'ⓒ': 'C', 'ⓓ': 'D', 'ⓔ': 'E',
        'ⓕ': 'F', 'ⓖ': 'G', 'ⓗ': 'H', 'ⓘ': 'I', 'ⓙ': 'J',
        'Ⓐ': 'A', 'Ⓑ': 'B', 'Ⓒ': 'C', 'Ⓓ': 'D', 'Ⓔ': 'E',
        'Ⓕ': 'F', 'Ⓖ': 'G', 'Ⓗ': 'H', 'Ⓘ': 'I', 'Ⓙ': 'J',
        '©': 'C', '®': 'R',
    }
    for k, v in circled_map.items():
        s = s.replace(k, v)
    # Strip currency symbols
    s = re.sub(r'^[\$€£¥₹]+\s*', '', s)
    s = re.sub(r'\s*[\$€£¥₹]+$', '', s)
    # Strip common units
    s = re.sub(r'\s*(V|A|Ω|ohms?|volts?|amps?|kips|inches?|in\.?|ft\.?|m|cm|mm|kg|g|Mbps|Hz|kHz|MHz|W|kW|MW|N|kN|Pa|MPa|GPa|J|kJ|MJ|%|degrees?|°|rad|s|ms|μs)\s*$', '', s, flags=re.IGNORECASE)
    # Try to parse as number and normalize
    try:
        val = float(s.replace(',', ''))
        # Format without trailing zeros: 65.00 -> 65, 8.40 -> 8.4
        if val == int(val):
            return str(int(val))
        return f"{val:g}"
    except (ValueError, OverflowError):
        return s.lower().strip()


def _parse_expected_alternatives(expected: str) -> list:
    """Parse expected answer into a list of acceptable alternatives.

    Handles formats like: "['24/7', '3.429', '3.43']" or malformed variants.
    """
    import ast, re
    s = expected.strip()
    # Try parsing as a Python list literal
    if s.startswith('['):
        try:
            alts = ast.literal_eval(s)
            if isinstance(alts, list):
                return [str(a) for a in alts]
        except (ValueError, SyntaxError):
            pass
        # Fallback: extract quoted/unquoted items from bracket notation
        items = re.findall(r"'([^']*)'?|\"([^\"]*)\"|([^\s,\[\]'\"]+)", s)
        alts = [next(g for g in groups if g).strip('[]') for groups in items if any(groups)]
        if alts:
            return alts
    return [s]


def _numeric_close(pred_str: str, exp_str: str, tolerance: float = 0.05) -> bool:
    """Check if two numeric strings are within tolerance of each other."""
    try:
        p = float(pred_str.replace(',', ''))
        e = float(exp_str.replace(',', ''))
        if e != 0:
            return abs(p - e) / abs(e) < tolerance
        else:
            return abs(p) < 0.01
    except (ValueError, OverflowError):
        return False


def _answers_match(predicted: str, expected: str, options: list = None) -> bool:
    """Compare predicted vs expected answers with normalization.

    For letter answers (A-J), does exact case-insensitive match.
    Also checks if predicted and expected letters map to identical option content.
    For numeric/open-ended answers, normalizes and compares with 5% tolerance.
    Supports multi-answer expected values like "['24/7', '3.429', '3.43']".
    """
    if not predicted or not expected:
        return False

    # Exact match (fast path, covers most MC questions)
    if predicted == expected:
        return True

    # Single-letter MC answers: case-insensitive
    if len(predicted.strip()) == 1 and len(expected.strip()) == 1:
        if predicted.strip().upper() == expected.strip().upper():
            return True
        # Check if both letters map to the same option content (duplicate options)
        if options:
            pred_letter = predicted.strip().upper()
            exp_letter = expected.strip().upper()
            letter_to_content = {}
            for opt in options:
                opt_str = str(opt).strip()
                if len(opt_str) >= 2 and opt_str[0].isalpha() and opt_str[1] in '.):':
                    letter = opt_str[0].upper()
                    content = opt_str[2:].strip().lstrip('.): ').strip()
                    letter_to_content[letter] = content
            pred_content = letter_to_content.get(pred_letter, "")
            exp_content = letter_to_content.get(exp_letter, "")
            if pred_content and exp_content and pred_content == exp_content:
                return True

    norm_pred = _normalize_answer(predicted)

    # Check against each acceptable alternative
    for alt in _parse_expected_alternatives(expected):
        norm_alt = _normalize_answer(alt)

        # Normalized string match
        if norm_pred == norm_alt:
            return True

        # Numeric tolerance (5%)
        if _numeric_close(norm_pred, norm_alt):
            return True

        # Handle fraction in predicted (e.g. "24/7") vs decimal in expected
        try:
            if '/' in norm_pred:
                parts = norm_pred.split('/')
                frac_val = float(parts[0]) / float(parts[1])
                if _numeric_close(str(frac_val), norm_alt):
                    return True
        except (ValueError, ZeroDivisionError, IndexError):
            pass

        # Handle fraction in expected vs decimal in predicted
        try:
            if '/' in norm_alt:
                parts = norm_alt.split('/')
                frac_val = float(parts[0]) / float(parts[1])
                if _numeric_close(norm_pred, str(frac_val)):
                    return True
        except (ValueError, ZeroDivisionError, IndexError):
            pass

    return False


def run_queue_worker(worker_id: int, job_queue: mp.Queue, config_dict: dict, stop_event):
    """Worker that pulls jobs from a shared queue."""
    worker_logger = logging.getLogger(f"worker_{worker_id}")

    # Stagger worker startup to avoid overwhelming the API with simultaneous requests
    stagger_delay = worker_id * 0.5
    time.sleep(stagger_delay)

    worker_logger.info(f"Worker {worker_id} starting (staggered {stagger_delay:.1f}s)...")

    # Load .env in worker process so per-agent keys are available
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

    from video_agent.agent import VideoUnderstandingAgent

    output_dir = Path(config_dict["output_dir"])
    live_question_file = output_dir / f"live_question_worker_{worker_id}.json"
    worker_results_file = output_dir / f"worker_results_{worker_id}.json"

    _agent_kwargs = dict(
        model=config_dict["model"],
        api_base=config_dict["api_base"],
        api_key=config_dict["api_key"],
        max_iterations=config_dict["max_iterations"],
        verbose=False,
        no_memory=config_dict.get("no_memory", False),
        orchestrator_model=config_dict.get("orchestrator_model"),
        max_ctx_tokens=config_dict.get("max_ctx_tokens"),
        raw_context=config_dict.get("raw_context", False),
        early_submit_gate=config_dict.get("early_submit_gate", True),
    )

    agent = VideoUnderstandingAgent(**_agent_kwargs)

    jobs_completed = 0
    
    while not stop_event.is_set():
        # Re-create agent if it was destroyed by a previous error
        if agent is None:
            try:
                agent = VideoUnderstandingAgent(**_agent_kwargs)
                worker_logger.info(f"Worker {worker_id}: Re-created agent after previous failure")
            except Exception as e:
                worker_logger.error(f"Worker {worker_id}: Failed to re-create agent: {e}")
                time.sleep(5)
                continue

        try:
            job = job_queue.get(timeout=2)
        except:
            if job_queue.empty():
                break
            continue
        
        if job is None:
            break
        
        video_data, q_idx = job["video_data"], job["question_idx"]
        video_id = video_data.get("video_id", "unknown")
        q_data = video_data.get("questions", [])[q_idx]
        question_id = q_data.get("question_id", f"{video_id}-{q_idx+1}")
        question_text = q_data.get("question", "")
        options = q_data.get("options", [])
        expected = q_data.get("answer", "")
        
        # Find video
        video_path = None
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            for name in [video_data.get("url", ""), video_id]:
                candidate = Path(config_dict["video_dir"]) / f"{name}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            if video_path:
                break
        
        if not video_path:
            worker_logger.warning(f"Worker {worker_id}: Video not found for {video_id}")
            continue
        
        worker_logger.info(f"Worker {worker_id}: Processing {question_id}")
        
        live_data = {
            "worker_id": worker_id,
            "question_id": question_id,
            "video_id": video_id,
            "question": question_text,
            "options": options,
            "expected": expected,
            "question_image": q_data.get("image_base64"),
            "status": "working",
            "trajectory": [],
            "token_usage": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        def step_callback(step_type, content, trajectory):
            live_data["trajectory"] = [
                {"step_type": s.step_type, "content": s.content, "timestamp": s.timestamp}
                for s in trajectory
            ]
            live_data["token_usage"] = agent.token_tracker.to_dict()
            try:
                with open(live_question_file, "w") as f:
                    json.dump(live_data, f, indent=2, default=str)
            except:
                pass
        try:
            with open(live_question_file, "w") as f:
                json.dump(live_data, f, indent=2, default=str)
        except:
            pass
        
        start_time = time.time()
        try:
            agent.step_callback = step_callback
            agent._current_question_id = question_id
            # Pass question images (base64) if present in dataset
            q_images = []
            if q_data.get("image_base64"):
                q_images = [q_data["image_base64"]]

            result = agent.run(
                video_path=str(video_path),
                question=question_text,
                options=options,
                question_images=q_images if q_images else None,
                task_type=q_data.get("task_type"),
                domain=video_data.get("domain"),
                sub_category=video_data.get("sub_category"),
            )

            elapsed = time.time() - start_time
            predicted = result.answer
            correct = _answers_match(predicted, expected, options=options) if predicted and expected else None
            if result.error:
                worker_logger.error(f"Worker {worker_id}: {question_id} agent error: {result.error}")
            if not result.success:
                worker_logger.warning(f"Worker {worker_id}: {question_id} failed (answer={predicted}, elapsed={elapsed:.1f}s)")
            
            result_data = {
                "worker_id": worker_id,
                "question_id": question_id,
                "video_id": video_id,
                "question": question_text,
                "options": options,
                "expected": expected,
                "question_image": q_data.get("image_base64"),
                "predicted": predicted,
                "correct": correct,
                "reasoning": result.reasoning,
                "elapsed_seconds": elapsed,
                "trajectory": [
                    {"step_type": s.step_type, "content": s.content, "timestamp": s.timestamp}
                    for s in result.trajectory
                ],
                "token_usage": result.token_usage or {},
                "error": result.error,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            
            # Append to per-worker results file (no locking needed — single writer)
            try:
                existing = []
                if worker_results_file.exists():
                    with open(worker_results_file, "r") as f:
                        existing = json.load(f)
                existing.append(result_data)
                with open(worker_results_file, "w") as f:
                    json.dump(existing, f, indent=2, default=str)
            except Exception as e:
                worker_logger.warning(f"Worker {worker_id}: Failed to write results: {e}")

            # Update live question status to completed
            live_data["status"] = "completed"
            live_data["predicted"] = predicted
            live_data["correct"] = correct
            live_data["elapsed_seconds"] = elapsed
            try:
                with open(live_question_file, "w") as f:
                    json.dump(live_data, f, indent=2, default=str)
            except:
                pass
            
            status = "✓" if correct else "✗"
            jobs_completed += 1
            worker_logger.info(
                f"Worker {worker_id}: {question_id} -> {predicted} "
                f"(exp: {expected}) {status} [{elapsed:.1f}s]"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            worker_logger.error(f"Worker {worker_id}: Error on {question_id}: {e}")
            # Reset agent so it gets re-created for the next question
            agent = None
            # Record failed questions so they appear in results
            result_data = {
                "worker_id": worker_id,
                "question_id": question_id,
                "video_id": video_id,
                "question": question_text,
                "options": options,
                "expected": expected,
                "predicted": None,
                "correct": False,
                "reasoning": None,
                "elapsed_seconds": elapsed,
                "trajectory": [],
                "token_usage": {},
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            try:
                existing = []
                if worker_results_file.exists():
                    with open(worker_results_file, "r") as f:
                        existing = json.load(f)
                existing.append(result_data)
                with open(worker_results_file, "w") as f:
                    json.dump(existing, f, indent=2, default=str)
            except Exception:
                pass
            jobs_completed += 1

    try:
        if live_question_file.exists():
            live_question_file.unlink()
    except:
        pass
    
    worker_logger.info(f"Worker {worker_id} finished. Completed {jobs_completed} jobs.")
    return jobs_completed



class ParallelEvaluator:
    """Orchestrates parallel evaluation with dynamic job queue."""
    
    def __init__(
        self,
        num_workers: int = 4,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 50,
        output_dir: str = "logs",
        no_memory: bool = False,
        orchestrator_model: Optional[str] = None,
        max_ctx_tokens: Optional[int] = None,
        raw_context: bool = False,
        early_submit_gate: bool = True,
    ):
        from .config import get_config
        cfg = get_config()
        self.num_workers = num_workers
        self.max_ctx_tokens = max_ctx_tokens
        self.raw_context = bool(raw_context)
        self.early_submit_gate = bool(early_submit_gate)
        self.model = model or cfg.main_agent.model
        self.orchestrator_model = orchestrator_model
        # Resolve API base: explicit arg > env var > config (None = official endpoint)
        self.api_base = (
            api_base
            or os.environ.get("MAIN_AGENT_API_BASE")
            or cfg.main_agent.api_base
            or ""
        )
        # Resolve API key: explicit arg > component env var > shared fallback keys
        self.api_key = (
            api_key
            or os.environ.get("MAIN_AGENT_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.no_memory = no_memory
    
    def run(self, dataset_path: str, video_dir: str, dashboard_port: int = 8080, run_id: str = None, max_questions: int = None):
        """Run parallel evaluation with dynamic job allocation."""
        with open(dataset_path, "r") as f:
            videos = json.load(f)

        # Build job queue - one job per question
        total_questions = 0
        jobs = []
        for video_data in videos:
            questions = video_data.get("questions", [])
            for q_idx in range(len(questions)):
                jobs.append({"video_data": video_data, "question_idx": q_idx})
                total_questions += 1

        if max_questions and total_questions > max_questions:
            jobs = jobs[:max_questions]
            total_questions = max_questions
            logger.info(f"Capped to {max_questions} questions")
        
        logger.info(f"Starting parallel evaluation with {self.num_workers} workers")
        logger.info(f"Dataset: {len(videos)} videos, {total_questions} questions")
        logger.info(f"Dynamic job queue: workers will pull jobs as they complete")
        
        # Clean only live worker files and worker results, preserve other files
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["live_question_worker_*.json", "worker_results_*.json",
                        "live_question.json", "dashboard_questions.json"]:
            for f in self.output_dir.glob(pattern):
                f.unlink()

        # Archive any existing parallel_eval_*.json results so dashboard shows
        # the current live run, not a stale completed run. On name collision,
        # append a timestamp suffix so repeated runs with the same --run-id
        # do not silently overwrite prior trajectories.
        archive_dir = self.output_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in self.output_dir.glob("parallel_eval_*.json"):
            target = archive_dir / f.name
            if target.exists():
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                target = archive_dir / f"{f.stem}__{ts}{f.suffix}"
            f.rename(target)

        run_id = run_id or f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_name = Path(dataset_path).stem
        start_time = time.time()

        # Create shared job queue and populate it
        job_queue = mp.Queue()
        for job in jobs:
            job_queue.put(job)
        
        # Add poison pills for clean shutdown
        for _ in range(self.num_workers):
            job_queue.put(None)
        
        stop_event = mp.Event()
        
        # Config dict for workers
        config_dict = {
            "video_dir": video_dir,
            "output_dir": str(self.output_dir),
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "max_iterations": self.max_iterations,
            "no_memory": self.no_memory,
            "orchestrator_model": self.orchestrator_model,
            "max_ctx_tokens": self.max_ctx_tokens,
            "raw_context": self.raw_context,
            "early_submit_gate": self.early_submit_gate,
        }

        # Start aggregator thread
        aggregator_stop = threading.Event()
        aggregator_thread = threading.Thread(
            target=self._run_aggregator,
            args=(aggregator_stop, total_questions, start_time, run_id, dataset_name),
            daemon=True,
        )
        aggregator_thread.start()
        
        # Start worker processes
        processes = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=run_queue_worker,
                args=(i, job_queue, config_dict, stop_event)
            )
            p.start()
            processes.append(p)
        
        # Wait for all workers to finish
        for p in processes:
            p.join()
        
        elapsed = time.time() - start_time
        aggregator_stop.set()
        aggregator_thread.join(timeout=5)
        
        # Merge per-worker result files
        results = []
        for wf in sorted(self.output_dir.glob("worker_results_*.json")):
            try:
                with open(wf, "r") as f:
                    results.extend(json.load(f))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read {wf}: {e}")
            finally:
                try:
                    wf.unlink()  # Clean up per-worker files
                except OSError:
                    pass
        
        correct = sum(1 for r in results if r.get("correct") is True)
        answered = len([r for r in results if r.get("predicted")])
        failed = len([r for r in results if r.get("error") and not r.get("predicted")])
        accuracy = correct / answered if answered > 0 else 0

        summary = {
            "total_videos": len(videos),
            "total_questions": total_questions,
            "answered": answered,
            "failed": failed,
            "correct": correct,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
            "rate_seconds_per_question": elapsed / answered if answered > 0 else 0,
            "num_workers": self.num_workers,
            "model": self.model,
        }

        logger.info(f"Evaluation complete: {correct}/{answered} ({accuracy:.1%}) in {elapsed:.1f}s [failed={failed}]")
        
        output_file = self.output_dir / f"parallel_eval_{run_id}.json"
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")

        # Write aggregated error log
        error_log_file = self._write_error_log(run_id, summary, results)
        logger.info(f"Error log saved to {error_log_file}")

        # Clean up the per-worker intermediate files created by this run.
        # The run log and active_run.json are kept so the dashboard can still
        # serve status for the completed run.
        for pattern in ["live_question_worker_*.json", "worker_results_*.json",
                        "live_question.json", "dashboard_questions.json"]:
            for f in self.output_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass

        return {"summary": summary, "results": results}

    def _write_error_log(self, run_id: str, summary: dict, results: list) -> Path:
        """Write an aggregated log with summary + detailed error info."""
        log_file = self.output_dir / f"eval_{run_id}.log"

        lines = []
        lines.append(f"=== Evaluation: {run_id} ===")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model: {summary.get('model', 'N/A')}")
        lines.append(f"Workers: {summary.get('num_workers', 0)}")
        lines.append("")
        lines.append("--- Summary ---")
        lines.append(f"Total questions: {summary.get('total_questions', 0)}")
        lines.append(f"Answered: {summary.get('answered', 0)}")
        lines.append(f"Correct: {summary.get('correct', 0)}")
        lines.append(f"Failed: {summary.get('failed', 0)}")
        lines.append(f"Accuracy: {summary.get('accuracy', 0):.1%}")
        lines.append(f"Elapsed: {summary.get('elapsed_seconds', 0):.1f}s")
        lines.append(f"Rate: {summary.get('rate_seconds_per_question', 0):.1f} s/question")
        lines.append("")

        # Incorrect answers
        incorrect = [r for r in results if r.get("correct") is False]
        if incorrect:
            lines.append(f"--- Incorrect Answers ({len(incorrect)}) ---")
            for r in incorrect:
                lines.append(f"  {r.get('question_id')}: predicted={r.get('predicted')} expected={r.get('expected')} ({r.get('elapsed_seconds', 0):.1f}s)")
            lines.append("")

        # Failed questions with full error details
        failed = [r for r in results if r.get("error") and not r.get("predicted")]
        if failed:
            lines.append(f"--- Failed Questions ({len(failed)}) ---")
            for r in failed:
                lines.append(f"\n  Question: {r.get('question_id')} (video {r.get('video_id')})")
                lines.append(f"  Worker: {r.get('worker_id')}")
                lines.append(f"  Elapsed: {r.get('elapsed_seconds', 0):.1f}s")
                lines.append(f"  Error: {r.get('error', 'unknown')}")
            lines.append("")

        with open(log_file, "w") as f:
            f.write("\n".join(lines))

        return log_file

    def _run_aggregator(self, stop_event, total_questions, start_time, run_id, dataset_name):
        """Background thread that updates dashboard state."""
        dashboard_file = self.output_dir / "dashboard_questions.json"

        def _collect_worker_results() -> list:
            """Read all per-worker result files and merge."""
            results = []
            for wf in self.output_dir.glob("worker_results_*.json"):
                try:
                    with open(wf, "r") as f:
                        results.extend(json.load(f))
                except (json.JSONDecodeError, Exception):
                    pass
            return results

        def _write_dashboard():
            results = _collect_worker_results()

            answered = len([r for r in results if r.get("predicted")])
            correct = sum(1 for r in results if r.get("correct") is True)
            accuracy = correct / answered if answered > 0 else 0
            elapsed = time.time() - start_time
            rate = elapsed / answered if answered > 0 else None

            # Aggregate token usage across all completed questions
            agg_tokens, agg_by_agent = aggregate_token_usage(results)

            # Aggregate errors
            failed = [r for r in results if r.get("error") and not r.get("predicted")]
            error_counts = {}
            for r in failed:
                # Extract short error type from the full error string
                err = r.get("error", "")
                # Truncate to first line / first 120 chars for grouping
                short = err.split("\n")[0][:120]
                error_counts[short] = error_counts.get(short, 0) + 1
            # Live errors: only track workers that are stuck/crashed, not transient tool errors
            live_errors = []

            dashboard_data = {
                "meta": {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "model": self.model,
                    "answered": answered,
                    "total_questions": total_questions,
                    "correct": correct,
                    "failed": len(failed),
                    "accuracy": accuracy,
                    "rate_s_per_q": rate,
                    "elapsed_seconds": elapsed,
                    "num_workers": self.num_workers,
                    "token_usage": agg_tokens,
                    "token_usage_by_agent": agg_by_agent,
                    "error_summary": error_counts,
                    "live_errors": live_errors[-10:],  # Last 10 live errors
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                "questions": results,
            }

            with open(dashboard_file, "w") as f:
                json.dump(dashboard_data, f, indent=2, default=str)

        while not stop_event.is_set():
            try:
                _write_dashboard()
            except Exception as e:
                logger.debug(f"Aggregator error: {e}")
            stop_event.wait(timeout=2)

        # Final write after stop — catches the last question(s) that finished
        # between the last poll and workers joining
        try:
            _write_dashboard()
        except Exception as e:
            logger.debug(f"Aggregator final write error: {e}")
