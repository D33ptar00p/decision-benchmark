from __future__ import annotations

import httpx

from bench.adapters.questions import label_from_answer, question_for
from bench.gpu import free_ollama
from bench.schema import Case


class OllamaDecisionAdapter:
    """Ollama decision models (Nimble, TEV) via POST /v1/systemone (Jev-style System One API)."""

    def __init__(self, name: str, model_id: str, host: str = "http://localhost:11434",
                 keep_alive: str = "10m"):
        self.name = name
        self.model_id = model_id
        self.keep_alive = keep_alive
        self.host = host
        self._http = httpx.Client(base_url=host, timeout=None)

    def _post(self, state, question: dict, keep_alive: str | int) -> dict:
        body = {"model": self.model_id, "state": state, "keep_alive": keep_alive,
                "questions": {"q": question}}
        resp = self._http.post("/v1/systemone", json=body)
        if resp.is_error:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["answers"]["q"]

    def warmup(self) -> None:
        free_ollama(self.host, keep=self.model_id)  # nothing else may hold the GPU
        self.decide(Case("warmup", "hello", ["a", "b"], "a"))  # loads the model onto the GPU

    def decide(self, case: Case) -> str:
        answer = self._post(case.query, question_for(case), self.keep_alive)
        return label_from_answer(case, answer)

    def close(self) -> None:
        """Unload so the next model gets the whole GPU."""
        self._post("bye", {"type": "choice", "instructions": "-", "criteria": {"a": "", "b": ""}}, 0)
