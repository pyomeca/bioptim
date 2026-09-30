# Architecture of Bioptim (anim_arch.py, scene ArchitecturePath, 15 s)

## What it teaches
The path from the user's inputs to the `Solution`, one box per stage, each labelled with its real module (relative to
`bioptim/`) and key method: `OptimalControlProgram` (`_check_arguments_and_build_nlp`), `NonLinearProgram`
(one per phase, `declare_shooting_points`), `ConfigureProblem.initialize`, penalties (`_finalize_penalties` ->
`_declare_continuity`), `OptimizationVectorHelper` (`vector`, `bounds_vectors`, `init_vector`), solver interface
(`interface_utils.generic_solve`), `Solution` (`from_dict`, `decision_states`, `integrate`, `graphs`).
Beat 2 replays the same boxes with real numbers of a tiny pendulum OCP: 1 phase, nx = 4, nu = 2, decision vector
125 = 1 (time) + 84 (X, 21 nodes x 4) + 40 (U, 20 x 2), 80 continuity constraints (20 x 4), IPOPT status 0, cost 8.09.

## Commands
    E=/c/Users/micka/miniconda3/envs/captury_biobuddy; export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
    PYTHONPATH=. python docs/animations/generate_arch_data.py      # writes data/arch_sizes.npz (real solve, use_sx=True)
    cd docs/animations && manim render -qh anim_arch.py ArchitecturePath

## Caveats
- Names were checked by grep against this version; the panel shows a simplified constructor call, not a full example.
- The real call order inside `OptimalControlProgram.__init__` is more intricate (`_prepare_dynamics`,
  `_prepare_bounds_and_init`, `_declare_multi_node_penalties`, `_finalize_penalties`, `_prepare_vector_layout`); the
  boxes show the conceptual path, not every private step. Only the IPOPT interface file is named; Fatrop, SQP and
  Acados have sibling files in `interfaces/`.
- The decision-vector sizes are for one CONSTANT-control RK4 problem; collocation or other control types change them.

## Exercises
1. Change `N` to 40 in `generate_arch_data.py` and predict the new vector size before running it.
2. Switch to `OdeSolver.COLLOCATION(polynomial_degree=3)` and see which of the numbers changes.
3. Add a second phase (see `example_multiphase.py`) and check `len(ocp.nlp)`.
