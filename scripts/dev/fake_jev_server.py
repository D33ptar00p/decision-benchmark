#!/usr/bin/env python3
"""A tiny fake /v1/systemone server (standard library only), to test scripts/run_jev_harness.py.

  python scripts/dev/fake_jev_server.py --port 8777
  uv run python scripts/run_jev_harness.py fake --server fake=http://127.0.0.1:8777/v1/systemone --runs 1

It always answers: the first option for choices, P(yes)=0.7 for yes/no, and level 1 for scores.
The numbers are meaningless; it only proves the plumbing (all question types, all datasets).
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    def do_GET(self):  # /health
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):  # /v1/systemone
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = body["questions"]["q"]
        if q["type"] == "choice":
            answer = {"type": "choice", "choice": next(iter(q["criteria"]))}
        elif q["type"] == "noul":
            answer = {"type": "noul", "noul": 0.7}
        else:
            n = len(q["criteria"])
            answer = {"type": "score", "score": 1.0, "probabilities": {str(i): (0.8 if i == 1 else 0.1) for i in range(n)}}
        out = json.dumps({"model": "fake", "answers": {"q": answer}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
