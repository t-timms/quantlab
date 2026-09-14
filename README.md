# quantlab

A shared, recipe-driven CLI wrapping the quantization backends already used
across `kat-coder-nvfp4`, `ornith-nvfp4`, `bible-ai-assistant`, and the
MiniCPM5/Spark-X2.5 quants — ModelOpt, llm-compressor, and (new, 2026-09-14)
`advanced-gguf-quantizer` for native NVFP4-GGUF — behind one interface instead
of a fresh one-off Python script per model.

## Why wrap instead of reimplement

This project does **not** reimplement any quantization math, NVFP4/GPTQ/AWQ
kernels, or GGUF writers. Those already exist, are maintained by teams larger
than a solo operator (NVIDIA's ModelOpt team, RedHat/vLLM's llm-compressor
team, the llama.cpp/ggml maintainers), and are what every model in this
portfolio already ships on. Reimplementing them would be redoing well-covered
work instead of the actual differentiator (REAP pruning + rigorous eval on
top of existing toolchains). `quantlab` is an orchestration layer: it reads a
declarative recipe, calls the real tool for the chosen backend, tracks
resumable state, and runs the same coherence gate every project has hit the
need for independently.

## What it actually does

Three things every per-model script in this portfolio has needed and mostly
built ad hoc, each time:

1. **Recipe files instead of hardcoded scripts.** One `llmcompressor`/`modelopt`/`gguf`
   backend module, driven by a TOML recipe, instead of
   `quantize_kat_gptq.py`, `quantize_spark.py`, `quantize_modelopt.py`,
   `quantize_llmc.py` each reimplementing the same oneshot/mtq.quantize/export
   scaffolding with small variations.
2. **Resumable runs.** A `manifest.json` per project directory tracks which
   stage (quantize / gate) finished. A VM restart, OOM, or the sm_120
   CUDA-graph hangs this box has hit repeatedly no longer means starting a
   multi-hour job over — `quantlab run` just skips whatever's already marked
   done. The manifest is pinned to the recipe file's hash, so an edited recipe
   fails loudly instead of silently reusing stale-config state.
3. **A real coherence gate, not "did it exit 0."** Generalized directly from
   `minicpm-quant/coherence_check.py`: inspects token ids, `finish_reason`,
   and pad-token fraction — not "output looks non-empty." This is the exact
   check that would have caught zaya1 shipping a checkpoint that passed every
   intermediate metric and emitted nothing but pad tokens. The degenerate-
   detection logic is unit-tested against that exact failure shape (see
   `tests/test_gates.py::test_zaya1_style_pad_collapse_is_caught`) and the
   MiniCPM coding-LoRA repetition-loop regression, without needing a GPU to
   run the tests.

## Backends

| backend | wraps | status |
|---|---|---|
| `llmcompressor` | `llmcompressor.oneshot` (GPTQ/AWQ/AutoRound/FP8, NVFP4A16 scheme) | ported from `minicpm-quant/quantize_llmc.py`, same API calls |
| `modelopt` | `modelopt.torch.quantization` (`mtq.quantize` + `export_hf_checkpoint`) | ported from `minicpm-quant/quantize_modelopt.py`. The `nvfp4_act_headroom` cfg (ModelOpt 0.47's new activation-calibration algorithm) is included but **unverified** — see the caveat comment in `backends/modelopt_backend.py` before trusting it |
| `gguf` | `michaelw9999/advanced-gguf-quantizer` (security-vetted 2026-09-14 — see the private `local-llm-playbook` repo, PART 10) | shells out to the tool's own `run` subcommand; requires a pre-built BF16 GGUF as input, this backend does not do HF->GGUF conversion (neither does the wrapped tool — see its own `AGENTS.md`) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[llmcompressor]"   # or [modelopt] / [gate] as needed

quantlab recipe init --name kat-nvfp4a16-gptq --backend llmcompressor --out recipes/kat.toml
# edit recipes/kat.toml: model, output, calibration fields

quantlab run recipes/kat.toml --project ~/runs/kat-nvfp4a16-gptq
quantlab status ~/runs/kat-nvfp4a16-gptq
```

Re-running `quantlab run` on the same `--project` after an interruption skips
whatever already finished. `--skip-gate` runs quantize only. `quantlab gate
<checkpoint>` runs the coherence check standalone against an existing
checkpoint, no recipe needed.

## What this is not (yet)

- No REAP-pruning backend yet — pruning happens before this tool via
  `reap-cuda` directly, same as every existing project. Could become a fourth
  backend later; not started.
- No eval-harness integration beyond an optional `gate.eval_command` hook that
  runs an existing script (e.g. `eval_repeat.sh`) and gates on its exit code.
- The `gate` extra needs `vllm`, which needs a GPU-capable box — the
  quantize-side backends similarly need real GPU hardware and are not
  exercised by this repo's test suite (which covers recipe parsing, resumable
  state, and the gate's degenerate-detection logic — everything that doesn't
  require a GPU or model weights to verify).

## Status

First working version, 2026-09-14. Built but not yet run against a real
model end-to-end — validated so far: recipe parsing/round-trip, resumable
project state (including the crash-recovery and stale-recipe-rejection
paths), and the coherence-gate logic, all under a real `pytest` run (18/18
passing), not just written and assumed correct. See `ROADMAP.md`.
