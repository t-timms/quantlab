"""Recipe schema: the declarative config every backend reads instead of a
hardcoded one-off script. Fields are the union of what quantize_llmc.py,
quantize_modelopt.py, and advanced-gguf-quantizer's own recipe.toml actually
use in the existing per-model repos - not a speculative abstraction.
"""

from __future__ import annotations

import pathlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib  # 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

Backend = Literal["llmcompressor", "modelopt", "gguf"]


class CalibrationConfig(BaseModel):
    # llmcompressor: a single HF dataset id, e.g. "theblackcat102/evol-codealpaca-v1"
    # modelopt: one or more named built-in datasets, e.g. ["open_code_reasoning", "nemotron-sft-swe-v2"]
    # gguf: a local calibration text file path (llama-imatrix -f)
    dataset: str | list[str] | None = None
    samples: int = 512
    max_seq_length: int = 2048


class GateConfig(BaseModel):
    enabled: bool = True
    prompts: list[str] = Field(
        default_factory=lambda: [
            "Write a Python function that returns the nth Fibonacci number.",
            "Explain what a hash map is in two sentences.",
            "What does John 3:16 say?",
        ]
    )
    # matches coherence_check.py's degenerate() heuristic exactly - token ids +
    # finish_reason, not "output looks non-empty" (the ZAYA1 lesson)
    max_tokens: int = 128
    min_unique_ids: int = 4  # <= this many distinct token ids over min_len => degenerate
    min_len_for_check: int = 10
    max_pad_fraction: float = 0.5
    min_text_len: int = 5
    # optional: a shell command to run after the coherence gate passes, e.g.
    # your existing eval_repeat.sh. Exit 0 = promote, nonzero = fail the gate.
    eval_command: str | None = None


class LlmCompressorConfig(BaseModel):
    method: Literal[
        "gptq_nvfp4a16", "autoround_nvfp4a16", "awq_gptq_nvfp4a16", "fp8", "mixed_nvfp4_fp8"
    ]
    targets: str | list[str] = "Linear"
    ignore: list[str] = Field(default_factory=lambda: ["lm_head"])
    dampening_frac: float = 0.1


class ModelOptConfig(BaseModel):
    cfg: Literal["w4a16_nvfp4", "nvfp4_awq_lite", "int4_awq", "nvfp4_act_headroom"]
    export_format: Literal["hf"] = "hf"


class GgufConfig(BaseModel):
    profile: Literal["nvfp4", "nvfp4_mxfp6", "mxfp6-primary", "mxfp8"] = "nvfp4"
    target_bpw: float = 4.8
    mode: Literal["fast", "normal", "deep"] = "normal"
    # path to the advanced-gguf-quantizer checkout; the binary at
    # <tool_dir>/build/bin/advanced-gguf-quantizer must already be built
    tool_dir: str = "~/advanced-gguf-quantizer"
    # this backend requires an already-converted BF16 GGUF as input (the tool
    # itself does not convert HF checkpoints - see its README/AGENTS.md)
    bf16_gguf: str | None = None


class Recipe(BaseModel):
    name: str
    model: str  # local path or HF repo id (backend-dependent which it accepts)
    output: str
    backend: Backend
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    llmcompressor: LlmCompressorConfig | None = None
    modelopt: ModelOptConfig | None = None
    gguf: GgufConfig | None = None

    @field_validator("backend")
    @classmethod
    def _backend_lower(cls, v: str) -> str:
        return v.lower()

    def model_post_init(self, __context: object) -> None:  # pydantic v2 hook
        section = {"llmcompressor": self.llmcompressor, "modelopt": self.modelopt, "gguf": self.gguf}[
            self.backend
        ]
        if section is None:
            raise ValueError(
                f"recipe.backend = {self.backend!r} but recipe.{self.backend} is not set"
            )

    @property
    def output_path(self) -> pathlib.Path:
        return pathlib.Path(self.output).expanduser()

    @property
    def model_path(self) -> pathlib.Path:
        return pathlib.Path(self.model).expanduser()


def load_recipe(path: str | pathlib.Path) -> Recipe:
    p = pathlib.Path(path).expanduser()
    with p.open("rb") as f:
        data = tomllib.load(f)
    return Recipe.model_validate(data)


def save_recipe(recipe: Recipe, path: str | pathlib.Path) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        tomli_w.dump(recipe.model_dump(exclude_none=True), f)


def init_template(name: str, backend: Backend) -> Recipe:
    """Build a starting recipe for `quantlab recipe init`, with sensible
    per-backend defaults matching the existing repos' own scripts."""
    common = dict(name=name, model="~/models/<model-dir-or-hf-id>", output=f"~/models/{name}")
    if backend == "llmcompressor":
        return Recipe(
            **common,
            backend="llmcompressor",
            calibration=CalibrationConfig(dataset="theblackcat102/evol-codealpaca-v1", samples=512),
            llmcompressor=LlmCompressorConfig(method="gptq_nvfp4a16"),
        )
    if backend == "modelopt":
        return Recipe(
            **common,
            backend="modelopt",
            calibration=CalibrationConfig(dataset=["open_code_reasoning", "nemotron-sft-swe-v2"], samples=512),
            modelopt=ModelOptConfig(cfg="w4a16_nvfp4"),
        )
    if backend == "gguf":
        return Recipe(
            **common,
            backend="gguf",
            calibration=CalibrationConfig(dataset="data/calibration.txt"),
            gguf=GgufConfig(),
        )
    raise ValueError(f"unknown backend {backend!r}")
