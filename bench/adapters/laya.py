from __future__ import annotations

from bench.adapters.questions import label_from_answer, question_for
from bench.gpu import free_ollama
from bench.schema import Case


class LayaAdapter:
    """Laya (ModernBERT-based encoder decision model), in-process, one forward pass per case.

    API per the project's README: laya.load(<hf id>).predict(state, questions).
    The English checkpoint reads at most 512 tokens (typed-decisions: 1024).
    """

    def __init__(self, name: str, checkpoint: str, ollama_host: str = "http://localhost:11434"):
        self.name = name
        self.checkpoint = checkpoint
        self.ollama_host = ollama_host
        self._agent = None

    def _load(self):
        if self._agent is None:
            import laya  # deferred: optional dependency (pip install laya)

            self._agent = laya.load(self.checkpoint)
        return self._agent

    def warmup(self) -> None:
        free_ollama(self.ollama_host)
        self._load()
        self.decide(Case("warmup", "hello", ["a", "b"], "a"))

    def decide(self, case: Case) -> str:
        result = self._load().predict(case.query, {"q": question_for(case)})
        return label_from_answer(case, result["answers"]["q"])

    def close(self) -> None:
        import gc

        self._agent = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
