"""python -m memmoring_rag serve --data DIR --key-file FILE [--port 8013] [--budget-yuan 400]
python -m memmoring_rag stats --data DIR"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ledger import Ledger
from .server import App, serve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="memmoring_rag")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve", help="serve AML Add/Search/Health")
    run.add_argument("--data", type=Path, required=True, help="data directory")
    run.add_argument("--key-file", type=Path, required=True, help="file holding the AML API key")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8013)
    run.add_argument("--budget-yuan", type=float, default=400.0,
                     help="Add is refused once embedding spend reaches this many yuan")
    stats = commands.add_parser("stats", help="print the call ledger summary")
    stats.add_argument("--data", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(App(args.data, args.key_file.read_text().strip(), args.budget_yuan), args.host, args.port)
    else:
        print(json.dumps(Ledger(args.data / "calls.db").summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
