#!/usr/bin/env python3
"""Run one command and record portable elapsed time and peak RSS."""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    started = time.monotonic()
    child = subprocess.Popen(command)

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    returncode = child.wait()
    elapsed = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = (
        f"Elapsed wall clock seconds: {elapsed:.6f}\n"
        f"Maximum resident set size (kbytes): {usage.ru_maxrss}\n"
        f"Exit status: {returncode}\n"
    )
    Path(args.output).write_text(record)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
