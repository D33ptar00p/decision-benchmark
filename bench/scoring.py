from __future__ import annotations


def normalize(text: str) -> str:
    return text.strip().strip("\"'`.").strip().lower()


def match_option(raw: str, options: list[str]) -> str | None:
    """Map a model's raw answer onto one of the options, or None if it matches none."""
    cleaned = normalize(raw)
    by_norm = {normalize(o): o for o in options}
    if cleaned in by_norm:
        return by_norm[cleaned]
    # tolerate chatty answers: pick the longest option contained in the text
    hits = [o for n, o in by_norm.items() if n and n in cleaned]
    return max(hits, key=len) if hits else None


def is_correct(predicted: str | None, expected: str) -> bool:
    return predicted is not None and normalize(predicted) == normalize(expected)
