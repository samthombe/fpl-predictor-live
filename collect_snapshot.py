"""
Daily FPL snapshot collector (Track A).
Saves the RAW bootstrap-static payload so we accumulate a 2026-schema corpus
for training/inference. Run manually once to test, then schedule daily.
"""
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests

# Absolute paths derived from THIS file -> immune to Task Scheduler's CWD.
BASE_DIR = Path(__file__).resolve().parent
SNAP_DIR = BASE_DIR / "data" / "snapshots"
LOG_DIR = BASE_DIR / "logs"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://fantasy.premierleague.com/",
}


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG_DIR / "collector.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_once() -> None:
    r = requests.get(BASE_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()

    elements = data.get("elements", []) if isinstance(data, dict) else []
    events = data.get("events", []) if isinstance(data, dict) else []

    # Off-season / empty guard: don't write a useless 0-player snapshot.
    if not elements:
        log("SKIP: bootstrap returned 0 elements (off-season or blocked).")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = SNAP_DIR / f"{today}_bootstrap.json"

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload)

    unfinished = sum(1 for e in events if not e.get("finished", False))
    log(
        f"OK {out_path.name}: {len(elements)} elements, "
        f"{len(events)} events ({unfinished} pending), "
        f"{len(payload)/1024:.0f} KB"
    )


if __name__ == "__main__":
    try:
        collect_once()
    except Exception:
        log("ERROR:\n" + traceback.format_exc())
        sys.exit(1)