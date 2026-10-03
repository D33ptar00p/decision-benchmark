#!/usr/bin/env python3
"""Run the web app with an extra in-memory "fake" model, to try the UI without touching the GPU.

  uv run python scripts/dev/fake_ui_server.py --port 8020
  # then open http://127.0.0.1:8020 and tick "Fake model (UI test)"

The fake model answers correctly except for every 4th case (so squares come out green and orange)
and sleeps a few milliseconds per case.
"""
import argparse
import itertools
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import uvicorn  # noqa: E402

from bench import app as appmod  # noqa: E402
from bench import config  # noqa: E402


class FakeAdapter:
    name = "fake"

    def __init__(self):
        self.calls = itertools.count()

    def warmup(self):
        pass

    def decide(self, case):
        time.sleep(0.01 + 0.002 * len(case.options))
        if next(self.calls) % 4 == 3:  # every 4th answer is wrong
            return next(o for o in case.options if o != case.expected)
        return case.expected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8020)
    args = ap.parse_args()
    config.MODEL_TAGS["fake"] = "fake:latest"
    config.LABELS["fake"] = "Fake model (UI test)"
    config.FACTORIES["fake"] = FakeAdapter
    # make the app believe Ollama has the fake model, whatever is really installed
    appmod.ollama.Client = lambda host: types.SimpleNamespace(
        list=lambda: types.SimpleNamespace(models=[types.SimpleNamespace(model="fake:latest")]))
    uvicorn.run(appmod.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
