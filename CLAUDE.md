# bioptim

Optimal control framework for biomechanics, built on CasADi. Agent guidance lives in `agent-context/`:
the rules first, then the codebase map. Keep `agent-context/OVERVIEW.md` up to date when you find a wrong fact.

@agent-context/RULES.md
@agent-context/OVERVIEW.md

## Key commands

- Environment: `conda env create -f environment.yml && conda activate bioptim`
- Format: `black . -l120 --exclude "external/*"`
- One test, from the repository root: `pytest tests/shardN/test_<topic>.py -k <name> -x`
- Examples GUI: `python -m bioptim.examples`
