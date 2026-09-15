"""GGUF backend. Wraps the already-vetted `michaelw9999/advanced-gguf-quantizer`
binary (security review: 2026-09-14, see local-llm-playbook PART 10) rather
than reimplementing NVFP4 GGUF quantization - that binary is the only working
path to native NVFP4-GGUF right now (llama.cpp core has the type and the
Blackwell kernels but no quantize path yet).

Precondition this backend does NOT handle: converting an HF safetensors
checkpoint to a BF16 GGUF. Run `convert_hf_to_gguf.py` yourself first (a
separate llama.cpp checkout, per the tool's own AGENTS.md "Project
Boundaries") and point recipe.gguf.bf16_gguf at the result.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

from ..project import Project
from ..recipe import Recipe

STAGE = "quantize"


def _tool_binary(tool_dir: pathlib.Path) -> pathlib.Path:
    return tool_dir / "build" / "bin" / "advanced-gguf-quantizer"


def _run_logged(cmd: list[str], log, *, env: dict | None = None) -> None:
    log.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
    log.flush()
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)


def _ensure_prereq_artifact(
    binary: pathlib.Path,
    tool_recipe: pathlib.Path,
    subcommand: str,
    artifact_path: str,
    bin_dir: pathlib.Path,
    log,
) -> None:
    """imatrix and KLD-base are NOT auto-built by `run` (confirmed against the
    tool's own docs/advanced-gguf-quantizer-imatrix-kld.md, after an earlier
    reading of main.cpp's pipeline-script generation wrongly suggested they
    were) - generate each explicitly via the tool's own
    imatrix-command/kld-command, which prints the exact recommended
    llama-imatrix/llama-perplexity invocation. Per that doc: "Use the
    generated command as-is... Do not add context, batch, stride, or runtime
    scheduling overrides" - so this runs the printed command verbatim rather
    than reconstructing flags itself.
    """
    if pathlib.Path(artifact_path).exists():
        log.write(f"[quantlab] {subcommand}: {artifact_path} already exists, reusing\n")
        return
    printed = subprocess.run(
        [str(binary), subcommand, str(tool_recipe)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    log.write(f"[quantlab] {subcommand} -> {printed}\n")
    log.flush()
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    _run_logged(shlex.split(printed), log, env=env)


def _write_tool_recipe(recipe: Recipe, project: Project) -> pathlib.Path:
    cfg = recipe.gguf
    assert cfg is not None
    assert cfg.bf16_gguf, "recipe.gguf.bf16_gguf must point at an existing BF16 GGUF"

    calib = recipe.calibration.dataset
    assert isinstance(calib, str), "gguf backend needs calibration.dataset as a local text file path"

    toml_path = project.dir / "advanced-gguf-quantizer.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[io]",
                f'input = "{cfg.bf16_gguf}"',
                f'output = "{recipe.output_path / (recipe.name + "-" + cfg.profile + ".gguf")}"',
                "",
                "[calibration]",
                f'corpus = "{calib}"',
                f'imatrix = "{project.dir / "imatrix.dat"}"',
                "",
                "[evaluation]",
                f'bf16_reference = "{cfg.bf16_gguf}"',
                f'corpus = "{calib}"',
                f'kld_base = "{project.dir / "bf16.kld"}"',
                "",
                "[target]",
                f'precision_mode = "{cfg.profile}"',
                f"target_bpw = {cfg.target_bpw}",
                # the tool's recipe parser wants an integer here (confirmed by
                # trial: "invalid integer value: 12.0" on a float) - round
                # down, which is the conservative direction for a memory budget
                *([f"vram_gb = {int(cfg.vram_gb)}"] if cfg.vram_gb is not None else []),
                "",
                "[quantizer]",
                f'mode = "{cfg.mode}"',
                "",
                *(["[base]", f"threads = {cfg.threads}", ""] if cfg.threads is not None else []),
            ]
        )
    )
    return toml_path


def run(recipe: Recipe, project: Project) -> str:
    if project.skip_if_done(STAGE):
        return str(recipe.output_path)

    cfg = recipe.gguf
    assert cfg is not None

    tool_dir = pathlib.Path(cfg.tool_dir).expanduser()
    binary = _tool_binary(tool_dir)
    if not binary.exists():
        raise RuntimeError(
            f"{binary} does not exist - build the tool first: "
            f"cmake -S {tool_dir} -B {tool_dir}/build -DGGML_CUDA=ON && "
            f"cmake --build {tool_dir}/build -j 20"
        )

    recipe.output_path.mkdir(parents=True, exist_ok=True)
    tool_recipe = _write_tool_recipe(recipe, project)

    project.start(STAGE)
    log_path = project.log_path(STAGE)
    bin_dir = binary.parent
    try:
        with log_path.open("w") as log:
            _ensure_prereq_artifact(
                binary, tool_recipe, "imatrix-command", str(project.dir / "imatrix.dat"), bin_dir, log
            )
            _ensure_prereq_artifact(
                binary, tool_recipe, "kld-command", str(project.dir / "bf16.kld"), bin_dir, log
            )
            _run_logged(
                [str(binary), "run", str(tool_recipe), "--project", str(project.dir / "gguf_run"), "--yes"],
                log,
            )
    except subprocess.CalledProcessError as e:
        project.fail(STAGE, f"advanced-gguf-quantizer pipeline step exited {e.returncode} - see {log_path}")
        raise

    project.finish(STAGE, output=str(recipe.output_path), tool_recipe=str(tool_recipe))
    return str(recipe.output_path)
