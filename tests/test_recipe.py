import pytest

from quantlab.recipe import Recipe, init_template, load_recipe, save_recipe


def test_init_template_llmcompressor_roundtrip(tmp_path):
    recipe = init_template("kat-test", "llmcompressor")
    path = tmp_path / "recipe.toml"
    save_recipe(recipe, path)

    loaded = load_recipe(path)
    assert loaded.name == "kat-test"
    assert loaded.backend == "llmcompressor"
    assert loaded.llmcompressor is not None
    assert loaded.llmcompressor.method == "gptq_nvfp4a16"
    assert loaded.modelopt is None
    assert loaded.gguf is None


def test_init_template_modelopt_roundtrip(tmp_path):
    recipe = init_template("mini-test", "modelopt")
    path = tmp_path / "recipe.toml"
    save_recipe(recipe, path)
    loaded = load_recipe(path)
    assert loaded.modelopt.cfg == "w4a16_nvfp4"
    assert loaded.calibration.dataset == ["open_code_reasoning", "nemotron-sft-swe-v2"]


def test_init_template_gguf_roundtrip(tmp_path):
    recipe = init_template("bible-test", "gguf")
    path = tmp_path / "recipe.toml"
    save_recipe(recipe, path)
    loaded = load_recipe(path)
    assert loaded.gguf.profile == "nvfp4"
    assert loaded.gguf.tool_dir == "~/advanced-gguf-quantizer"


def test_backend_without_matching_section_rejected():
    with pytest.raises(ValueError, match="recipe.llmcompressor is not set"):
        Recipe(name="x", model="m", output="o", backend="llmcompressor")


def test_output_and_model_path_expand_user():
    recipe = init_template("x", "llmcompressor")
    recipe.output = "~/models/x"
    assert str(recipe.output_path).startswith(str(recipe.output_path.home()))
