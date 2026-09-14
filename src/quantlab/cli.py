#!/usr/bin/env python
"""quantlab - shared recipe-driven CLI for the quant pipeline.

  quantlab recipe init --name kat-nvfp4a16-gptq --backend llmcompressor --out recipes/kat.toml
  quantlab run recipes/kat.toml --project ~/runs/kat-nvfp4a16-gptq
  quantlab status ~/runs/kat-nvfp4a16-gptq
  quantlab gate ~/models/kat-nvfp4a16-gptq

Every run is resumable (manifest.json in --project) and quality-gated
(coherence check, same token-id/finish-reason discipline as
minicpm-quant/coherence_check.py) before being called done. See README.md.
"""

from __future__ import annotations

import argparse
import json
import sys

from .backends import get_backend
from .gates import GateConfig, run_coherence_gate
from .project import Project
from .recipe import init_template, load_recipe, save_recipe
from .report import write_report


def cmd_recipe_init(args: argparse.Namespace) -> int:
    recipe = init_template(args.name, args.backend)
    save_recipe(recipe, args.out)
    print(f"[quantlab] wrote template recipe -> {args.out}")
    print("[quantlab] edit model/output/calibration fields before running")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    project = Project(args.project, args.recipe)

    backend = get_backend(recipe.backend)
    print(f"[quantlab] {recipe.name}: backend={recipe.backend} project={project.dir}")
    output = backend.run(recipe, project)
    print(f"[quantlab] quantize stage -> {output}")

    gate_verdict = None
    if recipe.gate.enabled and not args.skip_gate:
        if project.skip_if_done("gate"):
            print("[quantlab] gate already recorded done - re-run with a fresh --project to re-gate")
        else:
            project.start("gate")
            try:
                gate_verdict = run_coherence_gate(output, recipe.gate)
                print(f"[quantlab] coherence gate: {gate_verdict.summary()}")
                for gen, degenerate in gate_verdict.results:
                    flag = "  <-- DEGENERATE" if degenerate else ""
                    print(f"  [{gen.prompt[:50]}] finish={gen.finish_reason} n_tok={len(gen.token_ids)}{flag}")
                if gate_verdict.passed:
                    project.finish("gate", summary=gate_verdict.summary())
                else:
                    project.fail("gate", gate_verdict.summary())
                    print(
                        "[quantlab] GATE FAILED - checkpoint is not promoted. "
                        "Fix and re-run, or inspect manually before trusting this output.",
                        file=sys.stderr,
                    )
            except ImportError:
                print(
                    "[quantlab] gate skipped: vllm not installed (pip install quantlab[gate])",
                    file=sys.stderr,
                )
    else:
        print("[quantlab] gate skipped (--skip-gate or recipe.gate.enabled=false)")

    report_path = write_report(recipe, project, gate_verdict)
    print(f"[quantlab] report -> {report_path}")
    return 0 if (gate_verdict is None or gate_verdict.passed) else 2


def cmd_status(args: argparse.Namespace) -> int:
    import pathlib

    manifest = pathlib.Path(args.project).expanduser() / "manifest.json"
    if not manifest.exists():
        print(f"[quantlab] no manifest at {manifest}", file=sys.stderr)
        return 1
    print(json.dumps(json.loads(manifest.read_text()), indent=2))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    cfg = GateConfig()
    verdict = run_coherence_gate(args.checkpoint, cfg)
    for gen, degenerate in verdict.results:
        flag = "  <-- DEGENERATE" if degenerate else ""
        print(f"[{gen.prompt[:50]}] finish={gen.finish_reason} n_tok={len(gen.token_ids)}{flag}")
        print(f"  text: {gen.text.strip()[:200]!r}")
    print(verdict.summary())
    return 0 if verdict.passed else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantlab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    recipe_p = sub.add_parser("recipe", help="recipe file management")
    recipe_sub = recipe_p.add_subparsers(dest="recipe_command", required=True)
    init_p = recipe_sub.add_parser("init", help="write a starting recipe template")
    init_p.add_argument("--name", required=True)
    init_p.add_argument("--backend", required=True, choices=["llmcompressor", "modelopt", "gguf"])
    init_p.add_argument("--out", required=True)
    init_p.set_defaults(func=cmd_recipe_init)

    run_p = sub.add_parser("run", help="run a recipe (resumable)")
    run_p.add_argument("recipe")
    run_p.add_argument("--project", required=True)
    run_p.add_argument("--skip-gate", action="store_true")
    run_p.set_defaults(func=cmd_run)

    status_p = sub.add_parser("status", help="print a project's manifest")
    status_p.add_argument("project")
    status_p.set_defaults(func=cmd_status)

    gate_p = sub.add_parser("gate", help="run the coherence gate standalone against a checkpoint")
    gate_p.add_argument("checkpoint")
    gate_p.set_defaults(func=cmd_gate)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
