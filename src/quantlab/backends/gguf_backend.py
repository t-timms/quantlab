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

import pathlib
import subprocess

from ..project import Project
from ..recipe import Recipe

STAGE = "quantize"


def _tool_binary(tool_dir: pathlib.Path) -> pathlib.Path:
    return tool_dir / "build" / "bin" / "advanced-gguf-quantizer"


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
                "",
                "[quantizer]",
                f'mode = "{cfg.mode}"',
                "",
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
    try:
        with log_path.open("w") as log:
            subprocess.run(
                [str(binary), "run", str(tool_recipe), "--project", str(project.dir / "gguf_run"), "--yes"],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as e:
        project.fail(STAGE, f"advanced-gguf-quantizer exited {e.returncode} - see {log_path}")
        raise

    project.finish(STAGE, output=str(recipe.output_path), tool_recipe=str(tool_recipe))
    return str(recipe.output_path)
