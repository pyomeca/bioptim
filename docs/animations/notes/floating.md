# Free floating base: reorientation without root torque (`FloatingReorient`, ~16 s)

## What it teaches
`TorqueFreeFloatingBaseBiorbdModel` (dynamics `TorqueFreeFloatingBaseDynamics`,
`bioptim/dynamics/state_space_dynamics/torque_dynamics_free_floating_base.py`) splits the state into
`q_roots, q_joints, qdot_roots, qdot_joints` and the control into `tau_joints` only: the torque of the root is
structurally zero, no `BiMapping` needed. Zero gravity (`gravity 0 0 0` in the bioMod), so the only way to rotate the trunk
is to move the arms: the arms are driven around a closed loop (start and end shape identical, `x_bounds["q_joints"][:, [0, -1]] = 0`)
and the trunk ends rotated by 0.80 rad (`x_bounds["q_roots"][2, -1] = 0.8`). Second part: the angular momentum about the CoM,
computed with `bio_model.angular_momentum()` (biorbd), is split into `L_root` (only `qdot_roots` non zero) and `L_joints`
(only `qdot_joints`); both reach +-10.7 kg m2/s, their sum is 7e-5 (tight axis).

## Real / not real
- Real: one IPOPT solve (`data/floating_reorient.npz`, IPOPT status 0, 94 iterations, N = 30, T = 2 s, RK4 with 3 steps per
  interval, `use_sx=True`); all curves and numbers in the video come from it. L, L_root, L_joints are computed at the nodes
  from the optimised states with biorbd. L is linear in qdot, so L = L_root + L_joints exactly.
- Model `models/floating_trunk_2arms.bioMod`: PLANAR variant (y-z plane, root = 2 translations + 1 rotation about x, two
  hinged arms). It is NOT the 3D case of `bioptim/examples/toy_examples/torque_driven_ocp/torque_driven_free_floating_base.py`
  (`trunk_and_2arm.bioMod`, 6-DoF root, somersault + twist), which was not run here.
- The body drawing is the model geometry (trunk 0.4 x 0.6 m, arms 0.7 m) linearly interpolated between the 30 nodes.
- Not shown: `qddot_roots` / `FreeFloatingBaseDynamics` with `qddot_root` do not exist in this version; the free floating
  dynamics available are `TorqueFreeFloatingBaseDynamics` (torque driven) and `StochasticTorqueFreeFloatingBaseDynamics`.
  (`BiorbdModel.forward_dynamics_free_floating_base` exists in `biorbd_model.py`, unused by this example.)

## Generate and render
```bash
E=/c/Users/micka/miniconda3/envs/captury_biobuddy
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
PYTHONPATH=. python docs/animations/generate_floating_data.py      # writes data/floating_reorient.npz
cd docs/animations && manim render -qh anim_floating.py FloatingReorient
```

## Caveats (honest)
- The residual |L| = 6.9e-5 is the solver/transcription tolerance (dynamics constraints satisfied to IPOPT tolerance,
  RK4 discretisation), not a physical drift; it is exactly zero for the continuous dynamics from L(0) = 0.
- Arm angles are bounded to +-1.5 rad in the interior nodes. Without that bound (first attempts, arm mass 2 kg) the
  optimiser made the arms windmill up to +-pi with saturated torques (100 Nm) and hit the iteration limit (status 1);
  heavier arms (4 kg, com 0.35 m) plus the bound converge cleanly. The trajectory is a local optimum found from a
  sinusoidal warm start (`InterpolationType.EACH_FRAME`), not a global optimum.
- The trunk angle is not monotone: it oscillates while the arms move and only the closed arm loop leaves the net 0.8 rad
  (geometric phase / holonomy: reorientation is possible for a cyclic change of shape because angular momentum is a
  non-integrable constraint).

## Exercises
1. Freeze one arm (`q_joints` bound to 0 for the right arm). The loop of the other arm alone in the (phi) line encloses no
   area: check that the net trunk rotation over a closed cycle becomes 0.
2. Change `TARGET` to 1.5 rad and compare the arm amplitude and torque needed; how does it scale?
3. Replace `MINIMIZE_CONTROL` by `ObjectiveFcn.Lagrange.MINIMIZE_ANGULAR_MOMENTUM` and add a non zero initial `qdot_roots`:
   what should the total L be then, and is it conserved?
