# External forces (anim_extforces.py)

## What it teaches
An external force known in advance (here a wind-like push on the hand, 15 N peak along -y) is described with
`ExternalForceSetTimeSeries(nb_frames=N)` + `add_translational_force(name, segment, values (3, N),
point_of_application_in_local=(3, N))`, given to the model (`TorqueBiorbdModel(path, external_force_set=fset)`) and to the
dynamics as `DynamicsOptions(numerical_data_timeseries={"external_forces": fset.to_numerical_time_series()})`.
The scene compares two real solves of the same reach (tracking of a minimum-jerk q, plus minimise tau): dashed = no
force, solid = with force, and plots the torque difference on its own axis. The force arrow is drawn on the hand.

## Commands (repo root, env captury_biobuddy, see task text for PATH)
    PYTHONPATH=. python docs/animations/generate_extforces_data.py      # -> data/extforces_arm.npz
    manim render -qh anim_extforces.py ExternalForces                   # from docs/animations

Files: anim_extforces.py, generate_extforces_data.py, data/extforces_arm.npz, models/extforces_arm.bioMod (planar
two-link arm, shoulder + elbow, rotations about x, hangs along -z, forearm attached with `RT 0 0 0 xyz 0 0 -0.3`).

## What is real
Both IPOPT solves (N = 30, T = 1.5 s, RK4, status 0, 7 and 8 iterations), the torques, the joint angles, and the stick
figures (biorbd marker positions of `elbow` and `hand` from the optimal q). The force profile is a chosen input
(Gaussian bump centred at 0.9 s, 15 N).

## Caveats
- The reach is enforced by a tracking objective (weight 100), not a hard constraint, so the trajectory also changes
  slightly with the force (arm lags/leads the reference). The torque difference is therefore not exactly -J^T F (the
  torque needed to cancel the force at fixed q); it also contains the change of motion. This was not checked against J^T F.
- With a pure "minimise tau + free path" objective, both problems give loopy swings (a poor motion), hence the tracking.
- Only the force set API of this version is shown (translational force in the global frame, point given in the segment
  frame). `ExternalForceSetVariables` (optimised forces) is not covered. The force is defined per shooting node (N values,
  piecewise constant over each interval).
- Torque plots show controls as piecewise constant per interval. Video length about 11.4 s.

## Exercises
1. Replace `add_translational_force` by `add_in_segment_frame(...)` (6 rows: torque then force) or `add_torque(...)` and
   compare the torques.
2. Raise `W_TRACK` to 1e4 and check that the difference of torques approaches the compensation of the force
   (compute J^T F with `bio_model.markers_jacobian()` at the tracked q).
3. Use `ExternalForceSetVariables` to let the optimiser choose the force and see what it does with a
   minimum-torque objective.
