# Multinode constraint and objective (`MultinodeLink`, ~16 s)

## What it teaches
A multinode penalty links variables of several nodes, possibly distant ones or in different phases. Here the last node
must return to the first (cyclic movement): the 31 nodes of the time grid are drawn, `Node.START` and `Node.END` are
highlighted and joined by an arc. Two ways to declare the link (API checked in this version, `bioptim/limits/multinode_*.py`):

```python
multinode_constraints = MultinodeConstraintList()
multinode_constraints.add(MultinodeConstraintFcn.STATES_EQUALITY,
                          nodes_phase=(0, 0), nodes=(Node.START, Node.END), key="all")
multinode_objectives = MultinodeObjectiveList()
multinode_objectives.add(MultinodeObjectiveFcn.STATES_EQUALITY,
                         nodes_phase=(0, 0), nodes=(Node.START, Node.END), weight=10, key="all")
OptimalControlProgram(..., multinode_constraints=multinode_constraints, multinode_objectives=multinode_objectives)
```
`nodes` accepts `Node.START/MID/PENULTIMATE/END` or integer indices (e.g. `nodes=(0, 15)`); `nodes_phase` must have the same
length. `CONTROLS_EQUALITY` links controls instead of states (not shown).

## Data (real, `data/multinode_pendulum.npz`)
Cart-pendulum (`pendulum.bioMod`), N = 30, T = 2 s, RK4, translation actuated, rotation passive, start q = 0 (qdot free),
`TRACK_STATE` q_rot = 1 rad at `Node.MID`, minimise the integral of tau^2. Three IPOPT solves (all status 0):

| solution | norm(x_END - x_START) | cost | iterations |
|---|---|---|---|
| no link | 15.25 (cart ends 2.5 m away, qdot gap ~10) | 2.25 | 87 |
| `MultinodeObjectiveFcn.STATES_EQUALITY`, weight 10 | 0.090 | 6.10 | 66 |
| `MultinodeConstraintFcn.STATES_EQUALITY` | 3.4e-33 | 5.96 | 82 |

The gap is the norm over the full state (q and qdot, 4 values). The plot shows q only (cart position, pendulum angle);
the red bar is the q gap at `t = T` (dashed line: start value).

## Generate and render
```bash
# repo root, conda env captury_biobuddy on the PATH
PYTHONPATH=. python docs/animations/generate_multinode_data.py
# from docs/animations
manim render -qh anim_multinode.py MultinodeLink
```

## Caveats (honest)
- The "no link" case is a strawman: with a free end state and a torque-only cost, the optimiser lets the cart drift; its
  low cost is not comparable with the linked ones (the link adds a requirement).
- The soft link result depends on the weight (10 here); the residual 0.09 would shrink with a larger weight.
- Only a single phase is shown; the same syntax links nodes of different phases (`nodes_phase=(0, 2)`), see
  `examples/toy_examples/feature_examples/example_multinode_constraints.py`.
- More than 3 nodes in one phase needs `PhaseDynamics.ONE_PER_NODE` (see that example).

## Exercises
1. Replace `Node.END` by the index `15` (mid-horizon) and link it to `Node.START`: what happens to the rotation constraint?
2. Sweep the objective weight (1, 10, 100, 1000) and plot gap vs cost.
3. Use `MultinodeConstraintFcn.CONTROLS_EQUALITY` on the first and last nodes: is the state gap closed as well?
