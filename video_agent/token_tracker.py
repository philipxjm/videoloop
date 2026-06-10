"""Token usage tracking across all API calls.

Records per-call token usage broken down by agent type (main_agent, vlm,
transcript, summarizer) and iteration.  One TokenTracker
instance lives on each VideoUnderstandingAgent — parallel workers each
have their own agent, so no thread-safety concerns.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TokenUsage:
    """A single API call's token usage."""
    agent_type: str       # "main_agent", "vlm", "transcript", "summarizer"
    model: str
    input_tokens: int
    output_tokens: int
    iteration: int
    timestamp: float = field(default_factory=time.time)
    call_site: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker:
    """Accumulates token usage for a single question run."""

    def __init__(self):
        self._records: List[TokenUsage] = []
        self._current_iteration: int = 0

    def set_iteration(self, iteration: int):
        self._current_iteration = iteration

    def record(
        self,
        agent_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        iteration: Optional[int] = None,
        call_site: str = "",
    ):
        self._records.append(TokenUsage(
            agent_type=agent_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            iteration=iteration if iteration is not None else self._current_iteration,
            call_site=call_site,
        ))

    def reset(self):
        self._records = []
        self._current_iteration = 0

    # ---- Aggregation ----

    def total(self) -> Dict[str, int]:
        inp = sum(r.input_tokens for r in self._records)
        out = sum(r.output_tokens for r in self._records)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    def by_agent(self) -> Dict[str, Dict[str, int]]:
        agents: Dict[str, Dict[str, int]] = {}
        for r in self._records:
            if r.agent_type not in agents:
                agents[r.agent_type] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            agents[r.agent_type]["input_tokens"] += r.input_tokens
            agents[r.agent_type]["output_tokens"] += r.output_tokens
            agents[r.agent_type]["total_tokens"] += r.input_tokens + r.output_tokens
        return agents

    def by_iteration(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        iters: Dict[int, Dict[str, Dict[str, int]]] = {}
        for r in self._records:
            if r.iteration not in iters:
                iters[r.iteration] = {}
            if r.agent_type not in iters[r.iteration]:
                iters[r.iteration][r.agent_type] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            entry = iters[r.iteration][r.agent_type]
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["total_tokens"] += r.input_tokens + r.output_tokens
        return iters

    def iteration_summary(self, iteration: int) -> Dict[str, int]:
        inp = sum(r.input_tokens for r in self._records if r.iteration == iteration)
        out = sum(r.output_tokens for r in self._records if r.iteration == iteration)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    def to_dict(self) -> dict:
        return {
            "total": self.total(),
            "by_agent": self.by_agent(),
            "by_iteration": {
                str(k): v for k, v in sorted(self.by_iteration().items())
            },
            "call_count": len(self._records),
        }


def aggregate_token_usage(records):
    """Sum per-record ``token_usage`` dicts into ``(total, by_agent)``.

    Each record is expected to carry ``record["token_usage"]`` in the shape
    produced by ``TokenTracker.to_dict()`` (``{"total": {...}, "by_agent": {...}}``).
    Shared by the parallel evaluator and the dashboard so the rollup logic
    lives in one place.
    """
    _FIELDS = ("input_tokens", "output_tokens", "total_tokens")
    total = {f: 0 for f in _FIELDS}
    by_agent: Dict[str, Dict[str, int]] = {}
    for r in records:
        tu = r.get("token_usage") or {}
        t = tu.get("total") or {}
        for f in _FIELDS:
            total[f] += t.get(f, 0)
        for agent_type, counts in (tu.get("by_agent") or {}).items():
            slot = by_agent.setdefault(agent_type, {f: 0 for f in _FIELDS})
            for f in _FIELDS:
                slot[f] += counts.get(f, 0)
    return total, by_agent
