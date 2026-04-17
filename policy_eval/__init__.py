"""Policy evaluation framework.

Entry point + ``ScriptArguments`` live at ``evaluate_policy.py`` at the repo
root. This package holds the reusable building blocks:

- ``types``: Example / Benchmark / Evaluator / GenerationConfig / EvalContext.
- ``wandb_utils``: wandb init (with resume) and custom-step-axis logging.
- ``generation``: vLLM lifecycle + generation helpers.
- ``rewards``: shared reward-model loading and chosen-score caching.
- ``evaluators``: built-in Evaluator implementations.
- ``benchmarks``: built-in Benchmark implementations + registry.
- ``eval_utils``: orchestration helpers used by the entry point.

To add a new benchmark:
    1. Write a builder in ``benchmarks.py`` (or a new module).
    2. Register it in ``BENCHMARK_BUILDERS``.
    3. Pass ``--benchmarks <name>`` on the CLI.

To add a new evaluator:
    1. Add a class in ``evaluators.py`` implementing the ``Evaluator`` protocol.
    2. Attach it to one or more benchmarks via the benchmark builder.
"""
