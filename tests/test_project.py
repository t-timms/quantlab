import pytest

from quantlab.project import Project, StageStatus


def _make_recipe_file(tmp_path, content="name = 'x'\n"):
    p = tmp_path / "recipe.toml"
    p.write_text(content)
    return p


def test_new_project_all_stages_pending(tmp_path):
    recipe_path = _make_recipe_file(tmp_path)
    project = Project(tmp_path / "proj", recipe_path)
    assert project.status("quantize") is StageStatus.PENDING
    assert not project.is_done("quantize")


def test_finish_marks_done_and_skip_if_done_short_circuits(tmp_path, capsys):
    recipe_path = _make_recipe_file(tmp_path)
    project = Project(tmp_path / "proj", recipe_path)
    project.start("quantize")
    assert project.status("quantize") is StageStatus.RUNNING
    project.finish("quantize", output="/tmp/out")
    assert project.is_done("quantize")

    assert project.skip_if_done("quantize") is True
    assert "already done (resuming)" in capsys.readouterr().out


def test_fail_records_error_and_is_not_done(tmp_path):
    recipe_path = _make_recipe_file(tmp_path)
    project = Project(tmp_path / "proj", recipe_path)
    project.start("quantize")
    project.fail("quantize", "CUDA OOM")
    assert project.status("quantize") is StageStatus.FAILED
    assert not project.is_done("quantize")
    assert project.all_stages()["quantize"]["error"] == "CUDA OOM"


def test_resuming_reloads_prior_state_from_manifest(tmp_path):
    recipe_path = _make_recipe_file(tmp_path)
    project_dir = tmp_path / "proj"
    p1 = Project(project_dir, recipe_path)
    p1.finish("quantize", output="/tmp/out")

    # simulate a fresh process (e.g. after a VM restart) reopening the same project
    p2 = Project(project_dir, recipe_path)
    assert p2.is_done("quantize")
    assert p2.all_stages()["quantize"]["output"] == "/tmp/out"


def test_recipe_edit_after_partial_run_is_rejected(tmp_path):
    recipe_path = _make_recipe_file(tmp_path, "name = 'x'\n")
    project_dir = tmp_path / "proj"
    Project(project_dir, recipe_path).start("quantize")

    recipe_path.write_text("name = 'x'\nextra = 'changed'\n")
    with pytest.raises(RuntimeError, match="different recipe"):
        Project(project_dir, recipe_path)
