# Reading the decision vector (anim_vector.py)

## What it teaches
IPOPT sees one flat vector. Bioptim builds it as `dt | X | U | algebraic | parameters` (`bioptim/optimization/vector_layout.py`):
- `VectorLayout.index_map` maps `(phase, "states"/"controls"/"algebraic_states", node)` and `("global", "time"/"parameters")`
  to `(slice, n_columns)`. Real example here (pendulum, N = 3, RK4, 24 variables): X node 1 is `slice(5, 9)`.
- `ordering_strategy=OrderingStrategy.VARIABLE_MAJOR` (default): all X, then all U. `TIME_MAJOR`: x0 u0 x1 u1 ... X node 1
  moves to `slice(7, 11)`. Same 24 numbers, other positions (the cells move in the animation).
- With collocation (degree 3) every X node holds `n_states_decision_steps` = 4 columns (node + 3 collocation points): 16 numbers per
  node, 60 variables in total. They are filled column after column (Fortran order).
- Recommended reading is with the helpers (`sol.decision_states()`, `sol.decision_controls()`, `sol.parameters[...]`), which
  do not depend on the ordering. `index_map` is for understanding, not for indexing by hand.

## Scenes (about 17.6 s total)
- `VectorOrdering` (11 s): strip VARIABLE_MAJOR, highlight X1 + index_map code, TIME_MAJOR reorder, reading code.
- `VectorCollocation` (6.6 s): strip with the 4 columns per X node, zoom on X1 as a 4x4 grid, code.

## Commands (repo root, env captury_biobuddy)
    PYTHONPATH=. python docs/animations/generate_vector_data.py        # -> data/vector_layout.npz
    cd docs/animations
    manim render -qh anim_vector.py VectorOrdering
    manim render -qh anim_vector.py VectorCollocation

## What is real
Layout (`ocp.vector_layout.index_map`, `total_size`), bounds/initial guess vectors, and the optimal vector (`sol.vector`)
of four real OCPs (RK4 / COLLOCATION(degree 3) x VARIABLE_MAJOR / TIME_MAJOR). The generator asserts that
`layout.unstack(vec)` and `sol.decision_states()["q"][1]` agree, and that the two orderings hold the same numbers (sorted
vectors equal to 1e-14). All IPOPT solves converged (status 0).

## Caveats
- The problem has one dummy-ish parameter `max_tau` (peak torque, bounded by |tau|), so that the parameter block is not empty.
- RK4 with cold start (linear q guess) did not converge for N = 3 (status 1); it is warm-started from the collocation
  solution at the nodes. Costs differ between RK4 (21.6) and collocation (11.2): coarse N = 3 discretisations are different
  problems, and the trajectories are not meant to be pretty (rotation dips to -0.9 rad before reaching 1 rad).
- The algebraic block exists in `index_map` but has zero size here (RK4 and collocation without algebraic states): it is
  mentioned in a note, not drawn.
- Time is fixed, so `dt` (0.333) is in the vector but bounded to a single value.
- "dq0" = qdot0; names are cosmetic, taken from the state order q0 q1 dq0 dq1.

## Exercises
1. Build the OCP with 5 shooting nodes and print `ocp.vector_layout.total_size`; predict it first.
2. Change `polynomial_degree` to 4 and to 5: what does `nlp.n_states_decision_steps(1)` return, and where does X node 2 start?
3. Use `Solution.from_vector(ocp, v)` with a vector made by `OptimizationVectorHelper.init_vector(ocp)` and read the initial guess
   with `decision_states()` under both orderings.
