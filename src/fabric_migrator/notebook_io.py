from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import nbformat
from nbformat import NotebookNode


class NotebookValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NotebookLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_cells: int = 1_000


def load_notebook(data: bytes, limits: NotebookLimits) -> NotebookNode:
    if len(data) > limits.max_bytes:
        raise NotebookValidationError(
            f"Notebook exceeds the {limits.max_bytes // (1024 * 1024)} MB limit."
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NotebookValidationError("Notebook must be UTF-8 encoded.") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NotebookValidationError(
            f"Notebook is not valid JSON (line {exc.lineno}, column {exc.colno})."
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cells"), list):
        raise NotebookValidationError("Notebook JSON must contain a cells array.")
    if len(raw["cells"]) > limits.max_cells:
        raise NotebookValidationError(
            f"Notebook exceeds the {limits.max_cells} cell limit."
        )
    if raw.get("nbformat") != 4:
        raise NotebookValidationError("Only Jupyter notebook format 4 is supported.")
    try:
        notebook = nbformat.reads(text, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:
        raise NotebookValidationError(f"Notebook schema validation failed: {exc}") from exc
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type in {"code", "markdown", "raw"} and not isinstance(
            cell.source, str
        ):
            raise NotebookValidationError(f"Cell {index} has an invalid source value.")
    return copy.deepcopy(notebook)


def write_notebook(notebook: NotebookNode) -> bytes:
    nbformat.validate(notebook)
    return nbformat.writes(notebook, version=4).encode("utf-8")
