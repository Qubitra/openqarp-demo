"""``qubitra-openqarp-demo``: launch the dashboard, or open it as an editable notebook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def build_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "marimo", "edit" if args.edit else "run", str(APP)]
    if args.host is not None:
        command += ["--host", args.host]
    if args.port is not None:
        command += ["--port", str(args.port)]
    if args.headless:
        command.append("--headless")
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qubitra-openqarp-demo",
        description=(
            "Fujitsu's OpenQARP portfolio diversification, live on Qubitra's hosted simulator. "
            "Set QUBITRA_API_KEY (and QUBITRA_API_URL for a non-default deployment) first."
        ),
    )
    parser.add_argument("--edit", action="store_true", help="open the app as an editable notebook")
    parser.add_argument("--host", default=None, help="address to bind, e.g. 0.0.0.0 in a container")
    parser.add_argument("--port", type=int, default=None, help="port to serve on")
    parser.add_argument(
        "--headless", action="store_true", help="serve without opening a browser window"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("QUBITRA_API_KEY"):
        print(
            "QUBITRA_API_KEY is not set: the dashboard will ask for a key, and keeps it in "
            "memory only.",
            file=sys.stderr,
        )
    return subprocess.call(build_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
