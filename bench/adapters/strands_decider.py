from __future__ import annotations

import os

from bench.adapters.questions import label_from_answer, question_for
from bench.gpu import free_ollama
from bench.schema import Case

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class StrandsDeciderAdapter:
    """In-process Strands Decider (System One): one forward pass picks an option."""

    def __init__(self, name: str, checkpoint: str, device: str | None = None):
        self.name = name
        self.checkpoint = checkpoint
        self.device = device
        self._engine = None

    def _load(self):
        if self._engine is None:
            # heavy imports (torch) deferred so the app starts fast without this model
            import torch
            from strands_decider.infer import load_engine

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._engine = load_engine(self.checkpoint, device=device)
        return self._engine

    def warmup(self) -> None:
        free_ollama(OLLAMA_HOST)  # Ollama models hold VRAM the decider needs
        self._load()  # model load + first-call CUDA init, excluded from timing
        self.decide(Case("warmup", "hello", ["a", "b"], "a"))

    def decide(self, case: Case) -> str:
        from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

        wire = question_for(case)
        question = {"noul": NoulQuestion, "score": ScoreQuestion}.get(wire["type"], ChoiceQuestion)(**wire)
        resp = self._load().ask(case.query, {"q": question})
        return label_from_answer(case, resp.answers["q"].model_dump())

    def close(self) -> None:
        """Free GPU memory so the next model gets the whole card."""
        import gc

        import torch

        self._engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
