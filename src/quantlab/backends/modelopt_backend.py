"""NVIDIA ModelOpt backend. Generalized from minicpm-quant/quantize_modelopt.py
and kat_ab/quantize_kat_v2.py. Includes `nvfp4_act_headroom` (new in ModelOpt
0.47, confirmed in the real CHANGELOG.rst 2026-09-14) as a selectable cfg -
this is the activation-calibration lever the KAT/Ornith W4A4 A/B scrap
decision identified as untried, for use on the next new base rather than a
KAT/Ornith re-open.
"""

from __future__ import annotations

from ..project import Project
from ..recipe import Recipe

STAGE = "quantize"

# maps recipe.modelopt.cfg -> the actual mtq.*_CFG object name, resolved lazily
# (modelopt isn't a hard dependency of this module) since some of these only
# exist on modelopt >= 0.47
#
# CAVEAT on "nvfp4_act_headroom": the first three entries are copied directly
# from quantize_modelopt.py / quantize_kat_v2.py, which actually ran. This one
# is NOT verified the same way - the 0.47 CHANGELOG.rst describes it as a
# *calibration algorithm* ("anchors the scale to a low percentile... via
# `weight_scale_algorithm`") shipped as a recipe YAML
# (`modelopt_recipes/general/ptq/nvfp4_act_headroom-kv_fp8_cast.yaml`), which
# may mean it's not a flat `mtq.*_CFG` constant at all, unlike the other three.
# Check `mtq.NVFP4_ACT_HEADROOM_CFG` actually exists (or find the real
# attribute/recipe-loading call) against your installed modelopt version
# before trusting this path - the RuntimeError below will at least fail loud
# rather than silently quantizing with the wrong algorithm if the name is wrong.
_CFG_ATTR = {
    "w4a16_nvfp4": "W4A16_NVFP4_CFG",
    "nvfp4_awq_lite": "NVFP4_AWQ_LITE_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "nvfp4_act_headroom": "NVFP4_ACT_HEADROOM_CFG",  # UNVERIFIED - see caveat above
}


def run(recipe: Recipe, project: Project) -> str:
    if project.skip_if_done(STAGE):
        return str(recipe.output_path)

    cfg = recipe.modelopt
    assert cfg is not None  # enforced by Recipe.model_post_init

    project.start(STAGE)
    try:
        import torch  # noqa: PLC0415
        import modelopt.torch.quantization as mtq  # noqa: PLC0415
        from modelopt.torch.export import export_hf_checkpoint  # noqa: PLC0415
        from modelopt.torch.utils.dataset_utils import (  # noqa: PLC0415
            create_forward_loop,
            get_dataset_dataloader,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        attr = _CFG_ATTR[cfg.cfg]
        if not hasattr(mtq, attr):
            raise RuntimeError(
                f"modelopt.torch.quantization has no {attr!r} - this recipe's "
                f"cfg={cfg.cfg!r} needs a newer nvidia-modelopt than what's "
                "installed. Check `pip show nvidia-modelopt` against the "
                "CHANGELOG before assuming the recipe is wrong."
            )
        mtq_cfg = getattr(mtq, attr)

        dataset = recipe.calibration.dataset
        if isinstance(dataset, str):
            dataset = [dataset]
        assert dataset is not None, "modelopt backend needs calibration.dataset set"

        tok = AutoTokenizer.from_pretrained(recipe.model)
        model = AutoModelForCausalLM.from_pretrained(
            recipe.model, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()

        samples = recipe.calibration.samples
        half = samples // 2
        dl = get_dataset_dataloader(
            dataset_name=dataset,
            tokenizer=tok,
            num_samples=[half, samples - half] if len(dataset) == 2 else [samples],
            max_sample_length=recipe.calibration.max_seq_length,
            batch_size=1,
            device="cuda",
        )
        forward_loop = create_forward_loop(model=model, dataloader=dl)

        model = mtq.quantize(model, mtq_cfg, forward_loop)
        mtq.print_quant_summary(model)

        out = recipe.output_path
        out.mkdir(parents=True, exist_ok=True)
        export_hf_checkpoint(model, export_dir=str(out))
        tok.save_pretrained(str(out))
    except Exception as e:  # noqa: BLE001 - re-raised after recording; broad by design
        project.fail(STAGE, str(e))
        raise

    project.finish(STAGE, output=str(recipe.output_path), cfg=cfg.cfg)
    return str(recipe.output_path)
