from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

from . import __version__
from .config import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(prog="slidetwin", description="High-fidelity course PDF translation with Docling + an OpenAI-compatible model")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    translate = commands.add_parser("translate", help="Extract, translate, review, typeset and validate")
    translate.add_argument("source", type=Path)
    translate.add_argument("--config", type=Path, default=Path("config.local.toml"))
    translate.add_argument("--out", type=Path, required=True)
    translate.add_argument("--work", type=Path)
    translate.add_argument("--pages", help="Optional source pages, e.g. 2,7-9; entire document is still textual context")
    translate.add_argument("--no-preview", action="store_true")
    translate.add_argument("--force-extract", action="store_true")
    probe = commands.add_parser("probe", help="Check the configured model with one short translation")
    probe.add_argument("--config", type=Path, default=Path("config.local.toml"))
    args = parser.parse_args(argv)
    load_dotenv(args.config.resolve().parent/".env", override=False)
    try:
        settings = Settings.load(args.config.resolve())
        if args.command == "probe":
            from .client import ModelClient
            client = ModelClient(settings.provider)
            try:
                result = client.complete([{"role": "user", "content": "Translate into Chinese: A tensor is an organized collection of numbers. Return only the translation."}])
                print(result)
            finally:
                client.close()
        else:
            from .pipeline import run
            run(args.source, args.out, args.work or args.out.parent/(args.out.stem+".work"), settings,
                pages=args.pages, preview=not args.no_preview, force_extract=args.force_extract)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"SlideTwin: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

