"""Resumable run state. A crashed/interrupted multi-hour job (VM reboot, OOM,
the sm_120 CUDA-graph hangs you've hit repeatedly) resumes instead of
restarting - manifest.json is the source of truth for what already finished.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from enum import Enum
from typing import Any

MANIFEST_NAME = "manifest.json"

Stage = str  # "quantize" | "gate" | "eval" - kept as plain str, not a closed enum,
# so a backend can add its own stage names without touching this module.


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Project:
    """One project dir = one recipe run. Layout:
        <project>/manifest.json   - stage state, recipe hash, timestamps
        <project>/logs/<stage>.log
        <project>/report.json     - written by report.py once the run completes
    """

    def __init__(self, project_dir: str | pathlib.Path, recipe_path: str | pathlib.Path):
        self.dir = pathlib.Path(project_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.recipe_path = pathlib.Path(recipe_path).expanduser()
        self.manifest_path = self.dir / MANIFEST_NAME
        self._state = self._load_or_init()

    def _recipe_hash(self) -> str:
        return hashlib.sha256(self.recipe_path.read_bytes()).hexdigest()[:16]

    def _load_or_init(self) -> dict[str, Any]:
        recipe_hash = self._recipe_hash()
        if self.manifest_path.exists():
            state = json.loads(self.manifest_path.read_text())
            if state.get("recipe_hash") != recipe_hash:
                raise RuntimeError(
                    f"{self.manifest_path} was created from a different recipe "
                    f"(hash {state.get('recipe_hash')} != {recipe_hash}). "
                    "Use a fresh --project dir, or confirm the recipe edit was "
                    "intentional and delete manifest.json to start over."
                )
            return state
        return {
            "recipe_hash": recipe_hash,
            "recipe_path": str(self.recipe_path),
            "created_at": time.time(),
            "stages": {},
        }

    def _save(self) -> None:
        self.manifest_path.write_text(json.dumps(self._state, indent=2))

    def status(self, stage: Stage) -> StageStatus:
        s = self._state["stages"].get(stage)
        return StageStatus(s["status"]) if s else StageStatus.PENDING

    def is_done(self, stage: Stage) -> bool:
        return self.status(stage) is StageStatus.DONE

    def start(self, stage: Stage) -> None:
        self._state["stages"][stage] = {
            "status": StageStatus.RUNNING.value,
            "started_at": time.time(),
        }
        self._save()

    def finish(self, stage: Stage, **extra: Any) -> None:
        entry = self._state["stages"].setdefault(stage, {})
        entry.update(status=StageStatus.DONE.value, finished_at=time.time(), **extra)
        self._save()

    def fail(self, stage: Stage, error: str) -> None:
        entry = self._state["stages"].setdefault(stage, {})
        entry.update(status=StageStatus.FAILED.value, finished_at=time.time(), error=error)
        self._save()

    def log_path(self, stage: Stage) -> pathlib.Path:
        return self.dir / "logs" / f"{stage}.log"

    def all_stages(self) -> dict[str, Any]:
        return dict(self._state["stages"])

    def skip_if_done(self, stage: Stage) -> bool:
        """Call at the top of a stage runner. True = caller should return early
        (already done, resumable-run semantics); False = caller should run it."""
        if self.is_done(stage):
            print(f"[quantlab] {stage}: already done (resuming) - skipping")
            return True
        return False
