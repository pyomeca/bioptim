This document is made to guide AI contributions. 

Read this file first (the rules), then `OVERVIEW.md` (the map of the codebase). When prose and code disagree, the code and the tests win; fix the prose.

- General considerations
  - Environment preparation
    - If required, a test environment can be set up using the provided `environment.yml` file to set up the development environment: `conda env create -f environment.yml && conda activate bioptim`.
    - To run tests and format code, also install what CI installs: `conda install pytest pytest-mpl black -c conda-forge`.
    - ACADOS is optional. It is the `external/acados` submodule, installed with `external/acados_install_linux.sh` (or `external/acados_install_mac.sh`). Only the ACADOS tests need it (`tests/shard1/test_acados_interface.py`, the ACADOS case of `tests/shard1/test_mhe.py`).
  - Source of truth
    - Current tests are considered the source of truth for the codebase, i.e., if the tests pass, the code is assumed to be correct. Therefore (apart from creating new tests), unless explicitly confirmed, no changes to the existing tests should be made.
    - Expected values in tests are hard-coded numbers (cost, states, controls). If a test looks wrong, say so in the issue or the PR description instead of changing the expected value.

- Working
  - When developing new features or making changes, consider the impact on technical debt. The technical debt should be regularly assessed and addressed to maintain code quality and prevent long-term issues.
  - A file named `OVERVIEW.md` is available so you can organize and index the codebase effectively as you wish. Please note, this is a shared resource between agents. That said, do not hesitate to update it as needed.
    - If you find a wrong fact in it, fix it in the same PR. Cite file paths and symbol names, not line numbers (they drift).
  - Before changing core behavior, read the closest tests and the "Scientific invariants" section of `OVERVIEW.md`.
  - A new feature is expected to come with (see `docs/contributing.md`):
    - a minimal but meaningful example in `bioptim/examples/`;
    - a test that compares numerical values;
    - an export in `bioptim/__init__.py` if the class is public, and an update of the API section of `README.md` if the public API changes;
    - NumPy-style docstrings on every function, class and module; type aliases from `bioptim/misc/parameters_types.py` in annotations; plural names for variables that can be plural.
  - Scratch work goes in `bioptim/examples/sandbox/` (git-ignored). Do not commit binary files (results, figures, pickles); the only exceptions are assets the code or tests need, such as test reference data (`tests/plot_reference_images/`, `.npy` files in `tests/shard6/`) and model meshes, and only when explicitly required.
  - Do not edit `external/` (git submodules) or generated code (`c_generated_code/`).

- Validation
  - Format before each commit: `black . -l120 --exclude "external/*"`. CI checks this on every PR with an unpinned black; `docs/contributing.md` names black 26.5.
  - Run the closest test first, from the repository root: `pytest tests/shardN/test_<topic>.py -k <name> -x`. The suite has 75 test files with heavy parametrization; run the relevant file or shard, not all of `tests/` by default. One test name can expand to hundreds of cases (`-k test_pendulum` in shard2: 512 cases, about 5 minutes); list them with `--collect-only -q` and narrow `-k` with the parametrization id (e.g. `-k "test_pendulum and RK4 and SHARED"`).
  - For numerical or solver changes, compare more than one scalar: cost, solver status, constraint violation and problem dimensions. Compare against `pyomeca/bioptim` master in the same environment, and record the package versions (`conda list`). Call a failure a pre-existing baseline issue only after reproducing it on master.

- Commits
  - Each commit should represent a single logical change.
  - Use clear and concise commit messages.
  - Prefix commit messages with the type of change (e.g., `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`). An optional scope is welcome (`fix(collocation): ...`). Older history does not follow this rule; follow it anyway.
  - Keep commits small (`docs/contributing.md` suggests 1 to 10 lines when possible).

- Issues
  - If access are granted on GitHub (i.e., triage or more), tag the issue with the appropriate labels:
    - `bug` or `enhancement`: A bug has been identified in the code or the issue suggests an improvement or new feature for the code.
    - `quick fix` or `long fix` : Indicates whether the issue can be resolved quickly or requires a longer time to fix (a day or less).
    - `ai-exploration`: The issue uses AI to explore new features or improvements. A pull-request should be referenced if applicable.

- Pull requests (PR)
  - Open PRs against `master` of `pyomeca/bioptim`.
  - Open new PRs as GitHub draft PRs (`gh pr create --draft`). This should be the default state for new PRs unless they are fully ready for review. The test workflows skip draft PRs, which preserves CI time and tokens; only the black check runs. When the PR is ready, mark it with `gh pr ready`, which starts the tests. A `draft` *label* has no effect on CI.
  - Prefix the PR title with `[WIP]` while in progress, `[RTR]` when ready to review, but `[RTM]` when ready to merge is done by a reviewer(`docs/contributing.md`).
  - Fill the checklist of `docs/pull_request_template.md`. For a change that breaks the public API, write the description from `docs/breaking_change_template.md` (before/after migration example).
  - If an active issue exists related to the pull request, reference the issue in the PR description:
    - `closes #<issue_number>`: Indicates that the pull request resolves the specified issue.
    - `refs #<issue_number>`: Indicates that the pull request references the specified issue without closing it.
    - `depends on #<issue_number>`: Indicates that the pull request depends on the specified issue being resolved first.
  - If access are granted on GitHub (i.e., triage or more), tag the pull request with the appropriate labels:
    - `bug` or `enhancement`: The PR addresses a bug in the code or implements a new feature.
    - `quick review` or `long review`: The PR is expected to take short or long time to review for a human reviewer, respectively. Short reviews are typically completed within 15 minutes. 
    - `ai-exploration`: The PR uses AI to explore new features or improvements. This PR will not be reviewed by human, unless explicitly requested. Review request should only be request when necessary and justified.
