#!/usr/bin/env python3
"""Every job of every workflow has a job-level `timeout-minutes`.

A job without one runs until GitHub's 6-hour limit when a step hangs (a post step of
actions/checkout did, holding a runner of the shared pool), so a hang must end at a
bound derived from the job's own durations.

A job that calls a reusable workflow (`uses:` at job level) cannot have the key and is
skipped. Fails when a job lacks the key, and when no job was read.

usage: workflow_timeouts.py [workflows dir]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

JOB = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_-]*):\s*$")
KEY = re.compile(r"^    (timeout-minutes|uses):")


def jobs(text: str) -> dict[str, set[str]]:
    """job name -> the job-level keys of interest it has"""
    out: dict[str, set[str]] = {}
    in_jobs, job = False, None
    for line in text.splitlines():
        if re.match(r"^\S", line):
            in_jobs, job = line.startswith("jobs:"), None
            continue
        if not in_jobs:
            continue
        m = JOB.match(line)
        if m:
            job = m.group(1)
            out[job] = set()
            continue
        k = KEY.match(line)
        if job and k:
            out[job].add(k.group(1))
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / ".github" / "workflows"
    files = sorted(list(root.glob("*.yml")) + list(root.glob("*.yaml")))
    n, missing = 0, []
    for f in files:
        for name, keys in jobs(f.read_text(encoding="utf-8")).items():
            n += 1
            if "uses" not in keys and "timeout-minutes" not in keys:
                missing.append(f"{f.name}: job `{name}` has no timeout-minutes")
    for m in missing:
        print(m)
    if n == 0:
        print("error: no job read (compared nothing)", file=sys.stderr)
        return 1
    if missing:
        return 1
    print(f"ok: {n} jobs in {len(files)} workflows, each has timeout-minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
