# Choosing the dynamics of a model (anim_dynamics.py)

## What the scene teaches
A Bioptim model class fixes the dynamics, that is what is a **state** and what is a **control**. Beat 1 maps the classes of
`bioptim/models/biorbd/model_dynamics.py` (torque-based, muscle-based, model structure) with their states and controls and
the video of the series that shows each one. Beat 2 solves the SAME task with four torque-type dynamics and compares them.

Names in the map: read from REAL optimal control problems built (not solved) with each class for `TorqueBiorbdModel`,
`TorqueActivationBiorbdModel`, `TorqueDerivativeBiorbdModel`, `JointAccelerationBiorbdModel`, `MusclesBiorbdModel`,
`MusclesWithExcitationsBiorbdModel` (arm26), `TorqueFreeFloatingBaseBiorbdModel` and `MultiTorqueBiorbdModel`. The
holonomic (`q_u`, `qdot_u` / `tau`), variational (`q`, `lambdas` / `tau`) and stochastic (`q`, `qdot` / `tau`, `k`, `ref`,
`cov`, plus `a`, `c` or `cholesky_cov` depending on the `SocpType`) rows are read from the classes
(`bioptim/dynamics/state_space_dynamics/*.py`, `bioptim/dynamics/configure_variables.py`), not from a built problem.

Task of beat 2 (`models/dynamics_acrobot.bioMod` = `double_pendulum.bioMod` without meshes plus 20 N.m actuators):
hanging double pendulum at rest; the first joint (root, `q[0]`) is never actuated, only the elbow (`q[1]`, range +-pi/2)
is. Reach `q[0] = 1 rad` at rest (`qdot = 0`) at T = 4 s, N = 30, RK4 (5 steps), multiple shooting, IPOPT, default
initial guess. Each dynamics minimises the integral of the square of its own control (weight 1).

Numbers shown (printed by `generate_dynamics_data.py`, all IPOPT status 0):

| Dynamics | states (per node) | control | decision vector | IPOPT iterations | own cost | elbow torque peak (N.m) | integral of tau_2^2 dt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TorqueBiorbdModel` | q, qdot (4) | tau (2) | 185 | 35 | 41.88 | 6.14 | 41.88 |
| `TorqueActivationBiorbdModel` | q, qdot (4) | tau = activation (2) | 185 | 29 | 0.1047 | 6.14 | 41.88 |
| `TorqueDerivativeBiorbdModel` | q, qdot, tau (6) | taudot (2) | 247 | 60 | 231.03 | 6.25 | 67.73 |
| `JointAccelerationBiorbdModel` | q, qdot (4) | qddot_joints (1) | 155 | 14 | 210.68 | 7.45 | 95.07 |

The decision vector is `1 (time step) + (N + 1) x states + N x controls` (bioptim/optimization/vector_layout.py).
Largest difference of the first-joint angle to the torque solution: 6e-5 rad (activation), 0.64 rad (torque rate),
0.67 rad (acceleration).

## Commands
    E=/c/Users/micka/miniconda3/envs/captury_biobuddy; export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python docs/animations/generate_dynamics_data.py      # writes data/dynamics_compare.npz, data/dynamics_names.npz
    python docs/animations/render_series.py anim_dynamics.py DynamicsOverview --lang both --strict --quality 1080p30

## Honest caveats
- **The four costs are not comparable**: each is the integral of the square of a different quantity (N.m, activation,
  N.m/s, rad/s^2). Only the physical elbow torque tau_2 is common: it is the control (torque), `activation x 20 N.m`
  (activation), the torque state (derivative) or the inverse-dynamics torque of the accelerations (acceleration; the
  script checks that the torque of the root joint is zero). Peak and integral of tau_2^2 use tau_2 at the start of each
  interval (controls are constant per interval), not a higher-order quadrature.
- **Torque and activation are the same problem in other units**: the activation is tau / 20 N.m, the cost is
  divided by 20^2 (41.88 / 400 = 0.1047), and the motions agree within IPOPT tolerance. That is the point of the
  remark, not a coincidence. The activation model needs `actuator` blocks (Tmax) in the bioMod, hence the own model
  file. Bounds: activation in [-1, 1] (the "positive" and "negative" actuators both have Tmax = 20 N.m).
- **Torque rate**: `tau` becomes a state (bounded to +-20 N.m, fixed to 0 at t = 0, my choice), `taudot` the control (+-500
  N.m/s; the optimum uses at most 18 N.m/s). The problem has 250 equality constraints for 247 variables before fixed
  variables are removed: CasADi prints a warning "NLP is overconstrained", IPOPT still converges (status 0).
- **Joint acceleration**: only the non-root joints are controlled (`qddot_joints`, 1 control); the root acceleration is
  computed by `forward_dynamics_free_floating_base`. On `pendulum.bioMod` (cart-pole) `nb_root = 2 = nb_q`, so there are no
  joint controls: this dynamics needs a passive root, hence the double pendulum instead of the cart-pole.
- **Tried and not used** (nothing faked): with the torque model, reaching q[0] = 1.5 rad at rest (T = 3 s and T = 4 s),
  2 rad (T = 3 s) and the full swing-up to 3.14 rad (T = 4 s, N = 30 and N = 40) all ended with IPOPT status 1 (for
  1.5 rad in 3 s the message was "converged to a point of local infeasibility"): the elbow range +-pi/2 and the 20 N.m
  limit make the underactuated swing-up hard from the default guess. The task shown, 1 rad in 4 s, converges for all
  four. It is a small motion, chosen for that reason; no warm start or continuation is used (each solve starts from the
  default guess), and other local minima may exist.
- The elbow reaches its joint limits (+-pi/2) in all four solutions: the limits are active constraints.
- Muscle rows are for the default `with_residual_torque=False`. With `with_residual_torque=True`, `MusclesBiorbdModel` has the
  controls `tau` and `muscles` (checked in `dynamics_names.npz`).
- `StochasticTorqueBiorbdModel` is shown with the "Robust path constraint (SOCP)" video, which uses a custom stochastic
  mass-point dynamics, not this class; only the idea (stochastic variables as extra controls) is the same.
- The code panel shows a simplified version of `generate_dynamics_data.py` (`prepare_ocp`, lines 62-113): the class
  and `key` change per dynamics, bounds, initial guess and `DynamicsOptions` are omitted (`...`). The linked examples are
  the reference for the full calls.
- Length: about 32 s of content (with the layer, 1.6 x native speed), more than the 16-30 s target, because two beats are
  needed (a map and a comparison).

## Exercises
1. Change `weight=1` of `MINIMIZE_CONTROL` on `taudot` to 0.01 and 100 and watch the peak elbow torque and the
   angle difference to the torque solution.
2. Replace `key="qddot_joints"` by a penalty on `key="tau"` in `TorqueBiorbdModel` (bounds unchanged) and add
   `MINIMIZE_CONTROL` on `qddot_joints` in `JointAccelerationBiorbdModel` with a bound of +-5 rad/s^2: which one still converges?
3. Use `MusclesBiorbdModel` on `arm26.bioMod` (`static_arm.py`) and print `list(ocp.nlp[0].controls.keys())` with and
   without `with_residual_torque=True`.
