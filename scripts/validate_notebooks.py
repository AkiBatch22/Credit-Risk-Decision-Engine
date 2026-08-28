"""Static integrity checks for the five portfolio notebooks.

This intentionally does not execute notebooks or manufacture outputs. It checks
JSON structure, Python syntax, imports, duplicate code, execution metadata, and
Home Credit column-name references.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.components.feature_contract import PRODUCTION_ENGINEERED_FEATURES  # noqa: E402
from src.components.feature_engineering import (  # noqa: E402
    BENCHMARK_ONLY_ENGINEERED_FEATURES,
)


NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = tuple(
    NOTEBOOK_DIR / f"0{index}_{name}.ipynb"
    for index, name in enumerate(
        (
            "eda",
            "feature_engineering",
            "baseline_model",
            "model_comparison",
            "threshold_optimization",
        ),
        start=1,
    )
)
OUTDATED_PATTERNS = (
    "sklearn.cross_validation",
    ".fit_sample(",
    ".as_matrix(",
    "DataFrame.append(",
)
COLUMN_PATTERN = re.compile(
    r"\b(?:(?:AMT|DAYS|NAME|FLAG|CNT|EXT|SK_ID)[A-Z0-9_]*|TARGET)\b"
)


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def validate_notebooks() -> list[str]:
    dataset_path = ROOT / "data" / "raw" / "application_train.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset header unavailable: {dataset_path}")
    header = set(__import__("pandas").read_csv(dataset_path, nrows=0).columns)
    allowed_columns = header | set(PRODUCTION_ENGINEERED_FEATURES) | set(
        BENCHMARK_ONLY_ENGINEERED_FEATURES
    )
    summaries: list[str] = []

    for path in EXPECTED_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise AssertionError(f"{path.name}: expected nbformat 4")
        code_cells = [
            cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
        ]
        seen: set[str] = set()
        execution_counts: list[int] = []
        referenced_columns: set[str] = set()

        for index, cell in enumerate(code_cells, start=1):
            source = _source(cell).strip()
            compile(source, f"{path.name}:code-cell-{index}", "exec")
            if source in seen:
                raise AssertionError(f"{path.name}: duplicate code cell {index}")
            seen.add(source)
            for pattern in OUTDATED_PATTERNS:
                if pattern in source:
                    raise AssertionError(
                        f"{path.name}: outdated pattern {pattern!r} in code cell {index}"
                    )
            referenced_columns.update(COLUMN_PATTERN.findall(source))

            tree = ast.parse(source)
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if importlib.util.find_spec(module) is None:
                            raise ImportError(f"{path.name}: import not found: {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if importlib.util.find_spec(module) is None:
                        raise ImportError(f"{path.name}: import not found: {node.module}")

            count = cell.get("execution_count")
            outputs = cell.get("outputs", [])
            if count is None and outputs:
                raise AssertionError(
                    f"{path.name}: unexecuted code cell {index} contains outputs"
                )
            if isinstance(count, int):
                execution_counts.append(count)

        if execution_counts != sorted(execution_counts) or len(execution_counts) != len(
            set(execution_counts)
        ):
            raise AssertionError(f"{path.name}: execution counts are out of order")
        unknown_columns = sorted(referenced_columns.difference(allowed_columns))
        if unknown_columns:
            raise AssertionError(
                f"{path.name}: unknown Home Credit column references: {unknown_columns}"
            )
        summaries.append(
            f"{path.name}: {len(code_cells)} code cells; syntax/imports/columns/order OK; "
            f"executed cells={len(execution_counts)}"
        )
    return summaries


def main() -> None:
    for summary in validate_notebooks():
        print(summary)


if __name__ == "__main__":
    main()
