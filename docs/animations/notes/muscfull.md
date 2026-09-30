# muscfull: full muscle-driven reach versus the torque-driven reach

Files: `anim_muscfull.py`, `generate_muscfull_data.py`, `data/muscfull_arm.npz`.

## What it teaches
Two scenes (about 14 s). `MuscFullPaths`: same arm26 model (2 dof, 6 muscles), same reach A to B at rest at both ends
(N = 30, T = 0.8 s, RK4), solved once with `MusclesBiorbdModel(..., with_residual_torque=False)` and
`MINIMIZE_CONTROL` on `"muscles"` (integral of a^2, an effort proxy, not a metabolic cost: the library has none, only
`MINIMIZE_POWER`, which is mechanical power), once with `TorqueBiorbdModel` and `MINIMIZE_CONTROL` on `"tau"`. Hand
paths compared. `MuscFullActivations`: activation bars per muscle (with running peak marks), and the joint torque implied
by the muscles (`bio_model.muscle_joint_torque()`) against the torque of the direct solve.

## Commands (repo root, env captury_biobuddy)
    PYTHONPATH=. python docs/animations/generate_muscfull_data.py
    cd docs/animations; manim render -qh anim_muscfull.py MuscFullPaths
    manim render -qh anim_muscfull.py MuscFullActivations

## Real data
Both IPOPT solves reach status 0 (muscles 18 it, torques 11 it). Readouts (peak activation 0.65 on BICshort, effort,
path lengths, co-contraction count) are computed from the npz.

## Caveats
- Both problems carry a small smoothing term on qdot (weight 0.01) so that the solution is not chattering.
- No hand-path cost: both hand paths detour (1.25 m muscles, 0.87 m torques, straight line 0.38 m). The two costs are
  not comparable numbers (different units), so no "which is cheaper" claim is made.
- The muscle-implied torque includes what biorbd computes from the muscle state (activation, length, velocity, passive
  part); the torque of the two solutions differs also because the trajectories differ.
- Activations are piecewise constant (last node has no control).

## Exercises
1. Add a Lagrange `MINIMIZE_MARKERS` on the hand to make the paths straight; how do activations change?
2. Use `with_residual_torque=True` and penalise `tau`; do the muscles still do the work?
3. Change T to 0.4 s: which muscles saturate first?
