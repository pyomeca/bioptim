# Working on bioptim

`bioptim` is a biomechanics optimal-control framework built on CasADi and biorbd.
Its public API is intentionally broad (`bioptim/__init__.py`); source code and
tests are the authority when documentation and implementation disagree.

Before any non-trivial change:

1. Read `docs/architecture/LLM_CONTEXT.md` and the relevant source and tests.
2. Inspect Git with `git status --ignore-submodules=all`; use a clean worktree
   when the current one contains unrelated work.
3. Preserve units, signs, reference frames, dimensions, variable/bounds/target
   scaling, and the selected transcription grid.
4. Identify the affected backend and variants: MX/SX, RK/collocation,
   `PhaseDynamics`, control type, and solver adapter when applicable.

Use Conda for development (`environment.yml`, environment name `bioptim`).

Validation is proportional to the change:

- Run the closest test file or node first: `pytest tests/shardN/test_file.py`.
- For scientific or solver changes, add the relevant solver/grid/phase-dynamics
  coverage and compare feasibility, cost, status, and dimensions—not only one
  numerical scalar.
- Run `black . -l120 --exclude "external/*"` for formatting changes.
- Treat a CI failure as a baseline issue only after reproducing it on
  `origin/master` and recording the dependency versions.

Do not add runtime dependencies for documentation or static analysis. Keep
architecture changes incremental; `external/`, `c_generated_code/`, caches,
and generated solver artifacts are not source architecture.
