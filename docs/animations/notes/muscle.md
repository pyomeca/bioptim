# anim_muscle - muscle-driven reaching OCP

Scene `MuscleReaching` (~14.7 s, 1080p60). Teaches: switching from `TorqueBiorbdModel` to `MusclesBiorbdModel`
makes the muscle activations (6 for arm26) the controls; they are bounded in [0, 1] via `u_bounds["muscles"]`,
regularised by `ObjectiveFcn.Lagrange.MINIMIZE_CONTROL` (key="muscles") and the reach is a Mayer
`SUPERIMPOSE_MARKERS` objective (target vs COM_hand, weight 1000). Many activations sit on the bounds (bang-bang-like).

Data: REAL IPOPT solve of `bioptim/examples/toy_examples/muscle_driven_ocp/static_arm.py` (`prepare_ocp`, arm26,
30 shooting nodes, T = 0.5 s, RK4, constant controls): status 0, 37 iterations, cost 1.754, final hand-target distance 3.0 cm.
Arm and hand positions are computed from the optimal q with biorbd markers.

Generate / render (repo root, env captury_biobuddy):
    PYTHONPATH=. python docs/animations/generate_muscle_data.py     # writes data/muscle_arm.npz (~15 s)
    cd docs/animations && manim render -qh anim_muscle.py MuscleReaching

Caveats
- The torque-model line is shown as a code difference only; no torque-driven solve is run.
- The example keeps a residual torque (`with_residual_torque=True`, bound +-1 N.m, also minimised); peak |tau| is 0.56 N.m,
  so it is small but not zero. Activations, not excitations: no activation dynamics.
- Stick figure is the planar (x, y) projection; the reported 3.0 cm is the 3D distance (z offset included).
- The reach is not exact (weight 1000 is a soft penalty); the solution is a local optimum, not proven global.

Exercises
1. Change `weight` (e.g. 10, 1e5) and observe hand-target error vs total activation.
2. Use `MusclesWithExcitationsBiorbdModel` (see muscle_excitations_tracker.py) and compare with activations as controls.
3. Set `with_residual_torque=False` and check that the problem is still feasible.
