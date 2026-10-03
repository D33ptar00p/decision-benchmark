from __future__ import annotations

import importlib.util
import json
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bench import config
from bench.runner import run_benchmark, run_model_on_datasets
from bench.schema import load_dataset

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="decision-benchmark")
app.mount("/static", StaticFiles(directory=WEB), name="static")
_runs: dict[str, dict] = {}
_dataset_runs: dict[str, dict] = {}
_lock = threading.Lock()  # one benchmark at a time: latency needs the GPU to itself


class RunRequest(BaseModel):
    dataset: str
    models: list[str] = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=1000)


class DatasetRunRequest(BaseModel):
    model: str
    datasets: list[str] = Field(min_length=1, max_length=12)
    limit: int = Field(default=50, ge=1, le=1000)


def _check_option_limit(model: str, cases: list, dataset: str) -> None:
    limit = config.MAX_OPTIONS.get(model)
    widest = max((len(c.options) for c in cases), default=0)
    if limit and widest > limit:
        raise HTTPException(
            400, f"{model} supports at most {limit} options per case but {dataset} has {widest}; "
                 "regenerate it with fewer options (scripts/fetch_datasets.py, N_OPTIONS)")


def _preview(query: str | dict | list, limit: int = 200) -> str:
    """Short text for the UI tooltip; states may be structured JSON."""
    text = query if isinstance(query, str) else json.dumps(query, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def _dataset_path(name: str) -> Path:
    path = (config.DATASET_DIR / f"{name}.jsonl").resolve()
    if path.parent != config.DATASET_DIR.resolve() or not path.exists():
        raise HTTPException(404, f"unknown dataset {name!r}")
    return path


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/datasets")
def datasets_page() -> FileResponse:
    return FileResponse(WEB / "datasets.html")


@app.get("/api/datasets")
def datasets() -> list[dict]:
    return [{"name": p.stem, "cases": sum(1 for _ in p.open())}
            for p in sorted(config.DATASET_DIR.glob("*.jsonl"))]


@app.get("/api/models")
def models() -> list[dict]:
    try:
        pulled = {m.model for m in ollama.Client(host=config.HOST).list().models}
        ollama_up = True
    except Exception:
        pulled, ollama_up = set(), False
    out = [
        {"name": "strands-decision", "tag": config.STRANDS_CHECKPOINT, "available": True, "hint": ""},
        {"name": "laya", "tag": config.LAYA_CHECKPOINT,
         "available": importlib.util.find_spec("laya") is not None, "hint": "not installed: pip install laya"},
    ]
    for name, tag in config.MODEL_TAGS.items():
        exact = tag if ":" in tag else f"{tag}:latest"  # exact tag: tev1:0.8b is not tev1:latest
        installed = exact in pulled
        hint = f"not pulled: {tag}" if ollama_up else "ollama not running"
        out.append({"name": name, "tag": tag, "available": installed, "hint": hint})
    for m in out:
        m["label"] = config.LABELS.get(m["name"], m["name"])
    return out


@app.post("/api/runs")
def create_run(req: RunRequest) -> dict:
    unknown = [m for m in req.models if m not in config.FACTORIES]
    if unknown:
        raise HTTPException(400, f"unknown models: {unknown}")
    cases = load_dataset(_dataset_path(req.dataset), req.limit)
    for m in req.models:
        _check_option_limit(m, cases, req.dataset)
    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = {"models": req.models, "cases": cases, "started": False}
    return {"run_id": run_id, "cases": [{"id": c.id, "query": _preview(c.query)} for c in cases]}


@app.get("/api/runs/{run_id}/events")
def events(run_id: str) -> StreamingResponse:
    run = _runs.get(run_id)
    if run is None or run["started"]:
        raise HTTPException(404, "unknown or already-started run")
    run["started"] = True

    def stream() -> Iterator[str]:
        with _lock:
            adapters = [config.FACTORIES[m]() for m in run["models"]]
            for event in run_benchmark(adapters, run["cases"]):
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/dataset-runs")
def create_dataset_run(req: DatasetRunRequest) -> dict:
    if req.model not in config.FACTORIES:
        raise HTTPException(400, f"unknown model: {req.model!r}")
    names = list(dict.fromkeys(req.datasets))  # drop duplicates, keep order
    loaded = {}
    for name in names:
        cases = load_dataset(_dataset_path(name), req.limit)
        _check_option_limit(req.model, cases, name)
        loaded[name] = cases
    run_id = uuid.uuid4().hex[:12]
    _dataset_runs[run_id] = {"model": req.model, "datasets": loaded, "started": False}
    return {"run_id": run_id, "datasets": [
        {"name": n, "cases": [{"id": c.id, "query": _preview(c.query)} for c in cs]}
        for n, cs in loaded.items()]}


@app.get("/api/dataset-runs/{run_id}/events")
def dataset_run_events(run_id: str) -> StreamingResponse:
    run = _dataset_runs.get(run_id)
    if run is None or run["started"]:
        raise HTTPException(404, "unknown or already-started run")
    run["started"] = True

    def stream() -> Iterator[str]:
        with _lock:
            adapter = config.FACTORIES[run["model"]]()
            for event in run_model_on_datasets(adapter, run["datasets"]):
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
