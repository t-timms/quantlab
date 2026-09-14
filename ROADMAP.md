# Roadmap

Not a changelog (see `CHANGELOG.md`) — where this is headed.

## Done

- Recipe schema (pydantic) unifying the fields actually used across
  `quantize_llmc.py`, `quantize_modelopt.py`, and `advanced-gguf-quantizer`'s
  own `recipe.toml` — not a speculative abstraction, built from reading the
  real scripts first.
- Resumable project/manifest state, with a real test for the crash-recovery
  path (reopen a project after a "restart," confirm it doesn't redo a
  finished stage) and the stale-recipe-edit-rejection path.
- Coherence gate ported from `minicpm-quant/coherence_check.py`, with the
  degenerate-detection logic split out and unit-tested against the zaya1
  pad-collapse shape and the MiniCPM coding-LoRA repetition-loop shape,
  without needing a GPU.
- Three backends wired: `llmcompressor` (verified API calls, matches the real
  script), `modelopt` (verified API calls, except the new `nvfp4_act_headroom`
  cfg which is flagged unverified), `gguf` (wraps the already security-vetted
  `advanced-gguf-quantizer`).
- CLI (`recipe init` / `run` / `status` / `gate`) smoke-tested against the
  real installed package.

## Next

1. **Real first run, GPU-gated.** Bible-Assistant via the `gguf` backend —
   this is the same validation run already planned in `local-llm-playbook`
   PART 10 before native NVFP4-GGUF goes to kat-coder-nvfp4/ornith-nvfp4; this
   tool should be what runs it. First real test of the `llmcompressor` and
   `modelopt` backends against actual GPU hardware is still open too — nothing
   in this repo has touched a GPU yet.
2. Fix the `nvfp4_act_headroom` cfg attribute name/usage against the real
   installed ModelOpt 0.47 API before it's trusted — currently a documented
   guess, not a verified call.
3. A REAP-pruning backend (wraps `reap-cuda`), so the "one lab, one CLI" idea
   covers pruning too, not just quantization. Not started — no design work
   done yet either.
4. Eval-harness integration beyond the current `gate.eval_command` shell-out —
   e.g. calling `eval_repeat.sh`-style repeated-draw harnesses directly and
   parsing their Wilson-CI output into the run report, instead of just a
   pass/fail exit code.
5. Migrate one already-shipped project's script (most likely
   `minicpm-quant/quantize_llmc.py`, since it's the direct source for the
   `llmcompressor` backend) to actually call `quantlab` for its next run, as
   the real-world proof this isn't just parallel infrastructure nobody uses.

## Explicitly not planned

- Reimplementing NVFP4/GPTQ/AWQ math or GGUF writers. See README.md "Why wrap
  instead of reimplement."
