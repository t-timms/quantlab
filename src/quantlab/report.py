"""Run report - the same idea as advanced-gguf-quantizer's "run artifacts:
locked recipe, run log, manifest, quantization report", generalized across
backends so every quantlab run leaves the same shape of evidence trail, not
just whatever a one-off script happened to print.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

from .gates import GateVerdict
from .project import Project
from .recipe import Recipe


@dataclasses.dataclass
class RunReport:
    recipe_name: str
    backend: str
    output: str
    gate_summary: str | None
    gate_passed: bool | None
    stages: dict
    generated_at: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_report(
    recipe: Recipe,
    project: Project,
    gate_verdict: GateVerdict | None,
) -> pathlib.Path:
    report = RunReport(
        recipe_name=recipe.name,
        backend=recipe.backend,
        output=str(recipe.output_path),
        gate_summary=gate_verdict.summary() if gate_verdict else None,
        gate_passed=gate_verdict.passed if gate_verdict else None,
        stages=project.all_stages(),
    )
    path = project.dir / "report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
