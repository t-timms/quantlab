"""Backend contract. Every backend wraps an existing, proven tool
(llm-compressor, ModelOpt, advanced-gguf-quantizer) - quantlab does not
reimplement any quantization math or kernels itself. See README.md
"Why wrap instead of reimplement".
"""

from __future__ import annotations

from typing import Protocol

from ..project import Project
from ..recipe import Recipe


class Backend(Protocol):
    name: str

    def run(self, recipe: Recipe, project: Project) -> str:
        """Produce the quantized checkpoint at recipe.output_path and return
        it as a string path. Must be resumable: check
        project.skip_if_done('quantize') first and call project.start/finish/fail
        around the actual work."""
        ...
