"""Coherence quality gate. Generalizes minicpm-quant/coherence_check.py:
inspect token ids + finish_reason, not "output looks non-empty" - the exact
lesson from zaya1 shipping a checkpoint that passed every intermediate check
and emitted nothing but pad tokens.

Split into a pure, GPU-free `evaluate_generation()` (unit-testable with fake
token ids) and a `run_coherence_gate()` that does the real vLLM serving - so
the degenerate-detection logic itself has test coverage without a GPU.
"""

from __future__ import annotations

import dataclasses
import subprocess

from .recipe import GateConfig


@dataclasses.dataclass
class GenerationResult:
    prompt: str
    text: str
    token_ids: list[int]
    finish_reason: str | None
    pad_token_id: int | None = None


@dataclasses.dataclass
class GateVerdict:
    passed: bool
    results: list[tuple[GenerationResult, bool]]  # (result, degenerate?)

    def summary(self) -> str:
        bad = sum(1 for _, d in self.results if d)
        return "PASS" if bad == 0 else f"FAIL ({bad}/{len(self.results)} degenerate)"


def is_degenerate(gen: GenerationResult, cfg: GateConfig) -> bool:
    ids = gen.token_ids
    n = len(ids)
    uniq = len(set(ids))
    pad_frac = (
        ids.count(gen.pad_token_id) / max(n, 1)
        if gen.pad_token_id is not None
        else 0.0
    )
    text = gen.text.strip()
    return (
        (uniq <= cfg.min_unique_ids and n > cfg.min_len_for_check)
        or (pad_frac > cfg.max_pad_fraction and n > cfg.min_len_for_check)
        or len(text) < cfg.min_text_len
    )


def evaluate_generation(results: list[GenerationResult], cfg: GateConfig) -> GateVerdict:
    scored = [(r, is_degenerate(r, cfg)) for r in results]
    return GateVerdict(passed=all(not d for _, d in scored), results=scored)


def run_coherence_gate(checkpoint: str, cfg: GateConfig, *, max_model_len: int = 2048) -> GateVerdict:
    """Real gate: serve `checkpoint` on vLLM (enforce_eager, matching the
    established pattern for a fresh coherence check) and score its output.
    Requires the `gate` extra (vllm) installed - import is deferred so
    recipe/project code stays GPU-dependency-free.
    """
    from vllm import LLM, SamplingParams  # noqa: PLC0415 - intentionally lazy

    llm = LLM(
        model=checkpoint,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.max_tokens)
    msgs = [[{"role": "user", "content": p}] for p in cfg.prompts]
    outs = llm.chat(msgs, sp)

    results = [
        GenerationResult(
            prompt=p,
            text=o.outputs[0].text,
            token_ids=list(o.outputs[0].token_ids),
            finish_reason=o.outputs[0].finish_reason,
            pad_token_id=tok.pad_token_id,
        )
        for p, o in zip(cfg.prompts, outs)
    ]
    return evaluate_generation(results, cfg)


def run_eval_command(command: str, cwd: str | None = None) -> tuple[bool, str]:
    """Run an existing eval script (e.g. eval_repeat.sh) as the second half of
    the gate. Exit 0 = promote. Returns (passed, combined_output)."""
    proc = subprocess.run(
        command, shell=True, cwd=cwd, capture_output=True, text=True, check=False
    )
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, output
