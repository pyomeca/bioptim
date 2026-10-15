# bioptim

Optimal control framework for biomechanics, built on CasADi. Agent guidance lives in `agent-context/`.
Read, in order:

1. [agent-context/RULES.md](agent-context/RULES.md): rules for AI contributions (environment, validation, commits, issues, PRs).
2. [agent-context/OVERVIEW.md](agent-context/OVERVIEW.md): verified map of the codebase. Shared between agents; fix it when you find a wrong fact.

## Key commands

- Environment: `conda env create -f environment.yml && conda activate bioptim`
- Format: `black . -l120 --exclude "external/*"`
- One test, from the repository root: `pytest tests/shardN/test_<topic>.py -k <name> -x`
- Examples GUI: `python -m bioptim.examples`
