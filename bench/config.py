from __future__ import annotations

import os
from pathlib import Path

from bench.adapters.laya import LayaAdapter
from bench.adapters.ollama_decision import OllamaDecisionAdapter
from bench.adapters.strands_decider import StrandsDeciderAdapter

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets"
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Ollama tags: confirm with `ollama list` once the downloads finish.
MODEL_TAGS = {
    "nimble": os.environ.get("NIMBLE_MODEL", "nimble"),
    "tev": os.environ.get("TEV_MODEL", "tev1:latest"),
    "tev-0.8b": os.environ.get("TEV_SMALL_MODEL", "tev1:0.8b"),
}
LAYA_CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "convaiinnovations/laya")

# Strands Decider is a HF checkpoint run in-process (not via Ollama)
STRANDS_CHECKPOINT = os.environ.get(
    "STRANDS_DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v19"
)

FACTORIES = {
    "nimble": lambda: OllamaDecisionAdapter("nimble", MODEL_TAGS["nimble"], HOST),
    "tev": lambda: OllamaDecisionAdapter("tev", MODEL_TAGS["tev"], HOST),
    "tev-0.8b": lambda: OllamaDecisionAdapter("tev-0.8b", MODEL_TAGS["tev-0.8b"], HOST),
    "strands-decision": lambda: StrandsDeciderAdapter("strands-decision", STRANDS_CHECKPOINT),
    "laya": lambda: LayaAdapter("laya", LAYA_CHECKPOINT, HOST),
}

# Per-model cap on options per case (Ollama decision API: one letter per option)
MAX_OPTIONS = {"nimble": 26, "tev": 26, "tev-0.8b": 26}

# Display names shown in the UI (ids stay short and stable)
LABELS = {
    "strands-decision": "Strands Decider 2B",
    "nimble": "Nimble 9B",
    "tev": "TEV1 4B",
    "tev-0.8b": "TEV1 0.8B",
    "laya": "Laya 421M",
}
