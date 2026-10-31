#!/usr/bin/env python3
"""Task1 entrypoint for the shared paired GR00T + SONIC evaluator."""

from __future__ import annotations

from evaluation_runtime import (
    EvaluationTaskAdapter,
    make_default_adapter,
    parse_args,
    run_evaluation,
)


MANIFEST_ROW_COUNT = 100


def make_task_adapter() -> EvaluationTaskAdapter:
    """Return the Task1 adapter from the neutral runtime module."""

    return make_default_adapter()


def main() -> None:
    adapter = make_task_adapter()
    run_evaluation(parse_args(adapter), adapter)


if __name__ == "__main__":
    main()
