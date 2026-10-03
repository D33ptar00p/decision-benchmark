"""Wire format shared by the decision-model adapters (Jev "System One" question types)."""
from __future__ import annotations

from bench.schema import Case

YES, NO = "yes", "no"
THRESHOLD = 0.5  # P(yes) at or above this is answered "yes"


def question_for(case: Case) -> dict:
    """The question as the models' APIs take it: a choice, or a yes/no (noul)."""
    if case.kind == "noul":
        return {"type": "noul", "instructions": case.instructions}
    desc = case.descriptions or {}
    if case.kind == "score":  # ordered level descriptions; the answer is the level's index
        return {"type": "score", "instructions": case.instructions,
                "criteria": [desc.get(o, "") for o in case.options]}
    return {"type": "choice", "instructions": case.instructions,
            "criteria": {o: desc.get(o, "") for o in case.options}}


def label_from_answer(case: Case, answer: dict) -> str:
    """Turn a model's answer into the label the scorer compares ('yes'/'no' or an option)."""
    if case.kind == "noul":
        if "noul" not in answer:
            raise ValueError(f"no 'noul' probability in answer (got keys {sorted(answer)})")
        return YES if float(answer["noul"]) >= THRESHOLD else NO
    if case.kind == "score":
        # JevBench's rule: the most likely level, else the rounded expected score
        probs = answer.get("probabilities")
        if probs:
            return str(max(probs, key=lambda k: probs[k]))
        if "score" not in answer:
            raise ValueError(f"no score in answer (got keys {sorted(answer)})")
        return str(min(max(int(round(float(answer["score"]))), 0), len(case.options) - 1))
    return answer["choice"]
