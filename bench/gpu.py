from __future__ import annotations

import httpx


def free_ollama(host: str, keep: str | None = None) -> list[str]:
    """Unload every loaded Ollama model (except `keep`) so a run starts with a free GPU.

    Latency is only comparable when no other model is resident or competing for VRAM.
    """
    unloaded: list[str] = []
    try:
        loaded = httpx.get(f"{host}/api/ps", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        return unloaded  # Ollama not reachable: nothing to free
    for m in loaded:
        name = m["name"]
        if name == keep:
            continue
        httpx.post(f"{host}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
        unloaded.append(name)
    return unloaded
