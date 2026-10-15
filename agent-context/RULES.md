This document is made to guide AI contributions. 

- General considerations
  - Environment preparation
    - If required, a test environment can be set up using the provided `environment.yml` file to set up the development environment.
  - Source of truth
    - Current tests are considered the source of truth for the codebase, i.e., if the tests pass, the code is assumed to be correct. Therefore (apart from creating new tests), unless explicitly confirmed, no changes to the existing tests should be made.

- Working
  - When developing new features or making changes, consider the impact on technical debt. The technical debt should be regularly assessed and addressed to maintain code quality and prevent long-term issues.
  - A file named `OVERVIEW.md` is available so you can organize and index the codebase effectively as you wish. Please note, this is a shared resource between agents. That said, do not hesitate to update it as needed.

- Commits
  - Each commit should represent a single logical change.
  - Use clear and concise commit messages.
  - Prefix commit messages with the type of change (e.g., `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`).

- Issues
  - If access are granted on GitHub (i.e., triage or more), tag the issue with the appropriate labels:
    - `bug` or `enhancement`: A bug has been identified in the code or the issue suggests an improvement or new feature for the code.
    - `quick fix` or `long fix` : Indicates whether the issue can be resolved quickly or requires a longer time to fix (a day or less).
    - `ai-exploration`: The issue uses AI to explore new features or improvements. A pull-request should be referenced if applicable.

- Pull requests (PR)
  - If an active issue exists related to the pull request, reference the issue in the PR description:
    - `closes #<issue_number>`: Indicates that the pull request resolves the specified issue.
    - `refs #<issue_number>`: Indicates that the pull request references the specified issue without closing it.
    - `depends on #<issue_number>`: Indicates that the pull request depends on the specified issue being resolved first.
  - If access are granted on GitHub (i.e., triage or more), tag the pull request with the appropriate labels:
    - `bug` or `enhancement`: The PR addresses a bug in the code or implements a new feature.
    - `quick review` or `long review`: The PR is expected to take short or long time to review for a human reviewer, respectively. Short reviews are typically completed within 15 minutes. 
    - `ai-exploration`: The PR uses AI to explore new features or improvements. This PR will not be reviewed by human, unless explicitly requested. Review request should only be request when necessary and justified.
    - `draft`: The PR is a draft and not ready for review. This should be the default state for new PRs unless they are fully ready for review as it won't run the CI checks which preserves tokens.
