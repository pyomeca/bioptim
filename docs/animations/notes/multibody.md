# MultiBiorbdModel: two bodies, one OCP (anim_multibody.py)

## What the scene teaches
- `MultiTorqueBiorbdModel((path_a, path_b))` (bioptim/models/biorbd/model_dynamics.py, inherits `MultiBiorbdModel`)
  puts several independent biorbd models in the same phase.
- There is no `q_0` / `q_1` key: the variables keep their usual names (`"q"`, `"qdot"`, `"tau"`) and are **stacked**,
  model after model. `bio_model.variable_index("q", i)` gives the slice of model `i`
  (here body A: `range(0, 1)`, body B: `range(1, 3)`). The scene shows one real node `x_k = [q | qdot]`, `u_k = tau`
  read from the solution, cells coloured by model.
- The dynamics are block-diagonal (nothing in the dynamics couples the bodies). The coupling is a plain penalty:
  `ConstraintFcn.SUPERIMPOSE_MARKERS` at `Node.END` between `"A_tip"` and `"B_tip"`. Marker names are looked up over all
  models (`MultiBiorbdModel.marker_index`), so they must be unique across models (first match wins).

## Real data
`generate_multibody_data.py` solves the OCP with IPOPT (N = 30, T = 1.5 s, RK4, torque bounds +-60, objectives
MINIMIZE_CONTROL on tau + MINIMIZE_STATE on qdot, weights 1) and stores q, qdot, tau, the biorbd markers and the tip
distance in `data/multibody_results.npz`. The scene asserts that its planar forward kinematics matches the markers
computed by biorbd at every node. Models: `models/multibody_a.bioMod` (1 DoF link, pivot y = -0.8),
`models/multibody_b.bioMod` (2 DoF, pivot y = +0.8). Result: IPOPT status 0, 8 iterations; tip distance 1.61 m at t = 0
and about 2e-16 m at the last node.

## Commands
```
E=/c/Users/micka/miniconda3/envs/captury_biobuddy; export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
PYTHONPATH=. python docs/animations/generate_multibody_data.py           # from the repo root
cd docs/animations && manim render -qh anim_multibody.py MultiBody         # 1080p60
```

## Caveats
- Between nodes the drawing interpolates q linearly (nodes are 0.05 s apart), so it is a visualisation of the node
  values, not of the RK4 sub-steps. The curve of the tip distance is computed from that interpolated kinematics.
- The example shipped with bioptim (`toy_examples/torque_driven_ocp/example_multi_biorbd_model.py`) does not couple the models through markers; it
  uses `BiMappingList` to share torques. The marker constraint here is our own construction, following the
  README description of `MultiBiorbdModel`.
- Without the `qdot` objective the first solve (same problem) also converged but B made an unnecessarily large
  elbow swing (2.4 rad); the qdot term selects a smooth solution. Not a global-optimum claim.
- `MultiBiorbdModel` does not handle contacts, parameters nor external forces (see its constructor/`check_contacts`).
- `tau` has N values (no control at the last node), the scene shows "-" there.

## Exercises
1. Make body A also a double pendulum, then print `variable_index("q", i)` and `variable_index("markers", i)`.
   What changes in the stacked layout?
2. Replace `SUPERIMPOSE_MARKERS` by a constraint at `Node.MID` and check that the bodies now meet halfway.
3. Share one torque between the two elbows with `BiMappingList` (see `toy_examples/torque_driven_ocp/example_multi_biorbd_model.py`) and compare
   the cost.
