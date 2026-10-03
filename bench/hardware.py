"""Machine-state helpers used by the benchmark scripts.

Latency is only comparable when the machine is in the same state: on battery the GPU is
power-capped, and a hot laptop GPU throttles. These helpers read that state and wait for it.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

POWER_SUPPLY_DIR = Path("/sys/class/power_supply")


def ac_state() -> bool | None:
    """True/False for AC power; None if there is no AC supply entry (desktop, macOS, ...)."""
    for supply in sorted(POWER_SUPPLY_DIR.glob("A*")):
        try:
            return (supply / "online").read_text().strip() == "1"
        except OSError:
            continue
    return None


def gpu_state() -> dict | None:
    """Temperature (C), SM clock (MHz) and whether thermal throttling is active; None without a GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        temp, clock, throttle = [x.strip() for x in out.split(",")]
        return {"temp_c": int(temp), "clock_mhz": int(clock), "thermal_throttle": throttle == "Active"}
    except Exception:
        return None


def auto_device() -> str:
    """'cuda' if an NVIDIA GPU answers, else 'cpu'."""
    ok = shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    return "cuda" if ok else "cpu"


def wait_cooldown(max_c: int, timeout_s: int = 480) -> None:
    """Wait until the GPU is at or below `max_c` degrees (0 disables); give up after `timeout_s`."""
    if max_c <= 0 or gpu_state() is None:
        return
    start = time.time()
    while time.time() - start < timeout_s:
        state = gpu_state()
        if state is None or state["temp_c"] <= max_c:
            return
        time.sleep(5)
    print(f"  (GPU still above {max_c} C after {timeout_s}s; continuing)")


def wait_for_ac(poll_s: int = 15) -> None:
    """Block while the machine is on battery (returns at once if AC power or unknown)."""
    if ac_state() is False:
        print("On battery - waiting for AC power (timings are not comparable on battery)...")
        while ac_state() is False:
            time.sleep(poll_s)
