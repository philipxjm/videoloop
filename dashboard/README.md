# Dashboard

Live evaluation dashboard (FastAPI + static frontend). `agent_cli.py
parallel-eval` starts it automatically; to run it standalone:

```bash
pip install -e ".[dashboard]"
uvicorn dashboard.server:app --host 0.0.0.0 --port 8080
```

It discovers the active run via `logs/active_run.json` and shows per-worker
live questions, trajectories, token usage, and cost (rates from the
`pricing:` section of `configs/config.yaml`).

## Archiving runs as a static site

- `python dashboard/store_run.py --results logs/parallel_eval_<run>.json --dataset <dataset.json>`
  creates `dashboard/static/data/runs/<run_id>.json` and updates
  `dashboard/static/data/index.json`.
- `python dashboard/build_static.py --clean` copies `dashboard/static`
  (including data) into `docs/` for GitHub Pages hosting.

Data directories (`dashboard/static/data/`, `dashboard/data/`) are generated
and gitignored.
