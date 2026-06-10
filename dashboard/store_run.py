import argparse
import json
import time
from collections import Counter
from pathlib import Path

# Minimal helper to keep the dashboard data format consistent.


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_meta(dataset):
    meta = {}
    for video in dataset:
        vid = video.get("video_id")
        domain = video.get("domain")
        subcat = video.get("sub_category")
        for q in video.get("questions", []):
            qid = q.get("question_id")
            if not qid:
                continue
            meta[qid] = {
                "video_id": vid,
                "domain": domain,
                "sub_category": subcat,
                "task_type": q.get("task_type"),
                "question": q.get("question"),
                "expected": q.get("answer"),
            }
    return meta


def build_run_doc(results, dataset, run_id: str, title: str | None = None, created_at: str | None = None):
    meta = build_meta(dataset)

    counts_domain = Counter()
    correct_domain = Counter()
    counts_task = Counter()
    correct_task = Counter()
    questions_out = []
    correct_total = 0

    for item in results.get("results", []):
        qid = item.get("question_id")
        m = meta.get(qid, {})
        is_correct = bool(item.get("correct"))
        correct_total += is_correct
        domain = m.get("domain", "Unknown")
        task = m.get("task_type", "Unknown")
        counts_domain[domain] += 1
        correct_domain[domain] += 1 if is_correct else 0
        counts_task[task] += 1
        correct_task[task] += 1 if is_correct else 0
        questions_out.append(
            {
                "question_id": qid,
                "video_id": m.get("video_id"),
                "domain": domain,
                "sub_category": m.get("sub_category"),
                "task_type": task,
                "question": m.get("question"),
                "expected": m.get("expected"),
                "predicted": item.get("predicted"),
                "correct": is_correct,
                "reasoning": item.get("reasoning"),
                "trajectory": item.get("trajectory"),
            }
        )

    total_qs = len(results.get("results", []))
    accuracy = correct_total / total_qs if total_qs else 0.0

    created_ts = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    doc = {
        "run_id": run_id,
        "title": title or run_id,
        "created_at": created_ts,
        "model": results.get("model"),
        "dataset": results.get("dataset"),
        "summary": {
            "total_questions": total_qs,
            "correct": correct_total,
            "accuracy": accuracy,
            "elapsed_seconds": results.get("elapsed_seconds"),
        },
        "breakdown": {
            "domains": [
                {
                    "name": dom,
                    "total": total,
                    "correct": correct_domain[dom],
                    "accuracy": (correct_domain[dom] / total) if total else 0.0,
                }
                for dom, total in sorted(counts_domain.items())
            ],
            "tasks": [
                {
                    "name": task,
                    "total": total,
                    "correct": correct_task[task],
                    "accuracy": (correct_task[task] / total) if total else 0.0,
                }
                for task, total in sorted(counts_task.items())
            ],
        },
        "questions": questions_out,
    }
    return doc


def update_index(runs_dir: Path, index_path: Path):
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        data = load_json(path)
        summary = data.get("summary", {})
        runs.append(
            {
                "run_id": data.get("run_id") or path.stem,
                "title": data.get("title") or data.get("run_id") or path.stem,
                "model": data.get("model"),
                "dataset": data.get("dataset"),
                "accuracy": summary.get("accuracy"),
                "correct": summary.get("correct"),
                "total_questions": summary.get("total_questions"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "file": path.name,
            }
        )
    write_json(index_path, {"runs": runs})


def parse_args():
    base = Path(__file__).resolve().parent
    default_runs = base / "static" / "data" / "runs"
    default_index = base / "static" / "data" / "index.json"

    p = argparse.ArgumentParser(description="Convert an evaluation results JSON into dashboard run data.")
    p.add_argument("--results", required=True, type=Path, help="Path to evaluation results JSON")
    p.add_argument("--dataset", required=True, type=Path, help="Path to dataset JSON used for the run")
    p.add_argument("--run-id", dest="run_id", default=None, help="Identifier for this run (default: results filename stem)")
    p.add_argument("--title", default=None, help="Optional display title")
    p.add_argument("--runs-dir", type=Path, default=default_runs, help="Directory to write run JSON files")
    p.add_argument("--index", type=Path, default=default_index, help="Path to write index.json")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run JSON")
    return p.parse_args()


def main():
    args = parse_args()
    results = load_json(args.results)
    dataset = load_json(args.dataset)

    run_id = args.run_id or args.results.stem
    run_path = args.runs_dir / f"{run_id}.json"

    if run_path.exists() and not args.overwrite:
        raise SystemExit(f"Run file already exists: {run_path} (use --overwrite to replace)")

    doc = build_run_doc(results, dataset, run_id=run_id, title=args.title)
    write_json(run_path, doc)
    update_index(args.runs_dir, args.index)
    print(f"Wrote run: {run_path}")
    print(f"Updated index: {args.index}")


if __name__ == "__main__":
    main()
