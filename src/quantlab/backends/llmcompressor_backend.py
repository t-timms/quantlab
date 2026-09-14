"""llm-compressor backend. Generalized from minicpm-quant/quantize_llmc.py
(the same shape used, with small variations, across kat-coder-nvfp4,
ornith-nvfp4, and the Spark/MiniCPM quants) - one recipe field selects the
method instead of a --method CLI flag on a one-off script per model.
"""

from __future__ import annotations

from ..project import Project
from ..recipe import Recipe

STAGE = "quantize"


def _build_calib(tok, dataset: str, samples: int, max_seq_length: int):
    from datasets import Dataset, load_dataset  # noqa: PLC0415

    ds = load_dataset(dataset, split=f"train[:{samples * 3}]")
    rows = []
    for r in ds:
        instr = r.get("instruction") or ""
        out = r.get("output") or ""
        t = (instr + "\n\n" + out).strip()
        if len(t) < 40:
            continue
        rows.append({"text": t})
        if len(rows) >= samples:
            break
    d = Dataset.from_list(rows)
    return d.map(
        lambda b: tok(b["text"], truncation=True, max_length=max_seq_length),
        remove_columns=d.column_names,
    )


def run(recipe: Recipe, project: Project) -> str:
    if project.skip_if_done(STAGE):
        return str(recipe.output_path)

    cfg = recipe.llmcompressor
    assert cfg is not None  # enforced by Recipe.model_post_init

    project.start(STAGE)
    try:
        import torch  # noqa: PLC0415
        from llmcompressor import oneshot  # noqa: PLC0415
        from llmcompressor.modifiers.autoround import AutoRoundModifier  # noqa: PLC0415
        from llmcompressor.modifiers.quantization import (  # noqa: PLC0415
            GPTQModifier,
            QuantizationModifier,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(recipe.model)
        model = AutoModelForCausalLM.from_pretrained(
            recipe.model, torch_dtype=torch.bfloat16, device_map="cuda"
        )

        if cfg.method == "fp8":
            llmc_recipe = [
                QuantizationModifier(scheme="FP8_DYNAMIC", targets=cfg.targets, ignore=cfg.ignore)
            ]
            ds = None
        else:
            assert isinstance(recipe.calibration.dataset, str), (
                "llmcompressor backend needs calibration.dataset as a single HF dataset id"
            )
            ds = _build_calib(
                tok,
                recipe.calibration.dataset,
                recipe.calibration.samples,
                recipe.calibration.max_seq_length,
            )
            if cfg.method == "gptq_nvfp4a16":
                llmc_recipe = [
                    GPTQModifier(
                        scheme="NVFP4A16",
                        targets=cfg.targets,
                        ignore=cfg.ignore,
                        dampening_frac=cfg.dampening_frac,
                    )
                ]
            elif cfg.method == "autoround_nvfp4a16":
                llmc_recipe = [
                    AutoRoundModifier(scheme="NVFP4A16", targets=cfg.targets, ignore=cfg.ignore)
                ]
            elif cfg.method == "mixed_nvfp4_fp8":
                llmc_recipe = [
                    GPTQModifier(
                        scheme="NVFP4A16",
                        targets=["re:.*mlp.*"],
                        ignore=cfg.ignore,
                        dampening_frac=cfg.dampening_frac,
                    ),
                    QuantizationModifier(
                        scheme="FP8_DYNAMIC", targets=["re:.*self_attn.*"], ignore=cfg.ignore
                    ),
                ]
            elif cfg.method == "awq_gptq_nvfp4a16":
                from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415

                llmc_recipe = [
                    AWQModifier(scheme="NVFP4A16", targets=cfg.targets, ignore=cfg.ignore),
                    GPTQModifier(
                        scheme="NVFP4A16",
                        targets=cfg.targets,
                        ignore=cfg.ignore,
                        dampening_frac=cfg.dampening_frac,
                    ),
                ]
            else:  # pragma: no cover - Recipe's Literal type already restricts this
                raise ValueError(f"unknown llmcompressor method {cfg.method!r}")

        out = str(recipe.output_path)
        oneshot(
            model=model,
            dataset=ds,
            recipe=llmc_recipe,
            max_seq_length=recipe.calibration.max_seq_length,
            num_calibration_samples=recipe.calibration.samples if ds is not None else 1,
            output_dir=out,
        )
        tok.save_pretrained(out)
    except Exception as e:  # noqa: BLE001 - re-raised after recording; broad by design
        project.fail(STAGE, str(e))
        raise

    project.finish(STAGE, output=str(recipe.output_path), method=cfg.method)
    return str(recipe.output_path)
