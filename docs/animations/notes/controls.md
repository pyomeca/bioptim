# Control interpolation: ControlType (scene `ControlTypes`, ~13 s)

## What it teaches
The same pendulum swing-up (pendulum.bioMod, 20 shooting nodes, T = 1 s, RK4 with 5 steps, `MINIMIZE_CONTROL` on
`tau`, |tau| <= 100 on the translation) is solved with the three `control_type=` values of `OptimalControlProgram`:

| ControlType | tau(t) | nodes with a control | columns per node | decision vector | IPOPT cost | iterations |
|---|---|---|---|---|---|---|
| `CONSTANT` | staircase | 20 | 1 | 125 | 39.93 | 67 |
| `LINEAR_CONTINUOUS` | piecewise linear | 21 | 2 (1 at the last node) | 127 | 38.26 | 68 |
| `CONSTANT_WITH_LAST_NODE` | staircase | 21 | 1 | 127 | 39.93 | 67 |

All three solves converged (IPOPT status 0). Sizes, costs and curves are read from real `sol` objects.

## Commands
From the repo root (conda env `captury_biobuddy` on the PATH):
`PYTHONPATH=. python docs/animations/generate_controls_data.py` (writes `data/controls_types.npz`), then from
`docs/animations`: `manim render -qh anim_controls.py ControlTypes`.

## Honest caveats
- The extra 2 variables of the last two types are the tau values of the extra node (nb_tau = 2; the rotation torque
  is bounded to 0 but is still a variable). With `LINEAR_CONTINUOUS` the "2 columns" are the interpolation
  points (u_k, u_k+1), not 2 independent variables: the vector only grows by one node.
- `CONSTANT_WITH_LAST_NODE` gives the same trajectory and cost as `CONSTANT` here because the objective uses
  `Node.ALL_SHOOTING` and no penalty touches the last control, so IPOPT leaves it at its initial value (0). It is useful
  when something needs a control at node N (e.g. a constraint or objective on `Node.END` involving tau).
- The `LINEAR_CONTINUOUS` cost is lower, but the two costs are integrals of different parametrizations (the Lagrange
  term is integrated with the interpolated control), so it is an illustration, not a strict ranking. Only one
  discretization and one local solution per type were computed.
- The last-node torque of the linear solution (-24 N) is large: nothing penalizes it beyond the integral term.

## Exercises
1. Change `n_shooting` to 10 and 40: how do the vector sizes and the gap between the cost of CONSTANT and LINEAR_CONTINUOUS evolve?
2. Add `ObjectiveFcn.Mayer.MINIMIZE_CONTROL` at `Node.END` with `CONSTANT_WITH_LAST_NODE`: does the last node now matter?
3. Add a `MINIMIZE_CONTROL` with `derivative=True` (smooth torque) for each type. Which one profits most?
