from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_INSTRUCTIONS = "Which option best matches the input?"


@dataclass(frozen=True)
class Case:
    id: str
    query: str | dict | list  # the "state": plain text, or structured JSON
    options: list[str]
    expected: str
    instructions: str = DEFAULT_INSTRUCTIONS
    descriptions: dict[str, str] | None = None  # optional per-option descriptions
    category: str | None = None
    kind: str = "choice"  # "choice"; "noul" (yes/no, options ["yes", "no"]); "score" (options "0".."n-1")
    metric: str = "exact"  # "ordinal" adds MAE and QWK to the summary (score tasks)


@dataclass(frozen=True)
class Result:
    model: str
    case_id: str
    correct: bool
    latency_ms: float
    predicted: str | None
    expected: str
    error: str | None = None
    ordinal: bool = False  # counted in the summary's MAE / QWK

    def to_dict(self) -> dict:
        return asdict(self)


def load_dataset(path: Path, limit: int | None = None) -> list[Case]:
    cases: list[Case] = []
    with path.open() as f:
        for line in f:
            if limit is not None and len(cases) >= limit:
                break
            if line.strip():
                row = json.loads(line)
                cases.append(Case(
                    str(row["id"]), row["query"], row["options"], row["expected"],
                    row.get("instructions") or DEFAULT_INSTRUCTIONS,
                    row.get("descriptions"), row.get("category"), row.get("kind", "choice"),
                    row.get("metric", "exact"),
                ))
    return cases
