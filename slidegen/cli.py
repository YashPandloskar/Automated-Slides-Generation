"""Command line interface:  python -m slidegen run paper.pdf -o outputs/paper"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import rebuild_slides, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slidegen", description="Turn a scientific PDF into a PowerPoint deck.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline on a PDF")
    run.add_argument("pdf", type=Path)
    run.add_argument("-o", "--out", type=Path, help="output folder (default: outputs/<pdf name>)")
    run.add_argument("--no-images", action="store_true", help="skip the LLaVA image branch")
    run.add_argument("--chart-values", action="store_true",
                     help="print LLaVA-extracted chart values under figures (approximate; only when consistent with the caption)")

    slides = sub.add_parser("slides", help="rebuild the .pptx from an existing context.json (no models needed)")
    slides.add_argument("context", type=Path)
    slides.add_argument("-o", "--out", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            out = args.out or Path("outputs") / args.pdf.stem
            paths = run_pipeline(args.pdf, out, use_images=not args.no_images, show_chart_values=args.chart_values)
            print(f"\nDeck:       {paths['pptx']}\nContext:    {paths['context']}\nEvaluation: {paths['evaluation']}")
        else:
            out = args.out or args.context.with_name("slides.pptx")
            print(f"Deck: {rebuild_slides(args.context, out)}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
