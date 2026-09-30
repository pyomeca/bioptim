# Penalty on the derivative of a control (anim_deriv.py)

## What the scene teaches
Adding `Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=w)` on top of the plain
`MINIMIZE_CONTROL` smooths the torque. Cart-pendulum swing-up (N = 30, T = 1 s, RK4, `ControlType.LINEAR_CONTINUOUS`),
w = 0, 1, 10, 100. The plain solution (grey ghost) has a sharp dip near t = 0.47 s and a drop to -36 N on the last node;
the derivative penalty removes both. A second axis shows |dtau/dt| (finite difference of the nodes, clipped at 250).
Readouts computed from the data: exact integral of tau^2, peak |dtau/dt| (1154, 215, 171, 161 N/s) and IPOPT iterations.

## What derivative=True does in this version (bioptim/limits/penalty_option.py, `elif self.derivative`)
The penalty function is evaluated at the end and at the start of the interval and subtracted: f(u_end) - f(u_start).
For `ControlType.CONSTANT` and `CONSTANT_WITH_LAST_NODE`, `u_end = u_start`, so the term is exactly 0: the solution is
bit-identical (cost 40.2796 with w = 0 and w = 100, checked in the data). It only acts with `LINEAR_CONTINUOUS`.
`explicit_derivative` is set internally by the continuity constraints (penalty.py, constraints.py); it is not a
practical user option, and `derivative` and `explicit_derivative` cannot both be True. The other way to penalise the
rate, which works with constant controls, is `TorqueDerivativeBiorbdModel` (tau becomes a state, taudot the control) with
`MINIMIZE_CONTROL, key="taudot"`; not shown here.

## Commands
    E=/c/Users/micka/miniconda3/envs/captury_biobuddy; export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
    PYTHONPATH=. python docs/animations/generate_deriv_data.py      # writes data/deriv_pendulum.npz
    cd docs/animations && manim render -qh anim_deriv.py DerivativePenalty

## Caveats
- The plain LINEAR_CONTINUOUS solution is IPOPT-optimal (status 0) but its last-node torque (-36 N) barely affects the
  dynamics (only the last interval), which is why it is free to spike. That is a real feature of the plain problem.
- w = 1, 10, 100 are solved by continuation (warm start from the previous weight), all IPOPT status 0. Other starts may
  give other local minima. The integral of tau^2 is not monotone in w (45.2, 51.3, 55.0 for w = 1, 10, 100, 47.1 plain).
- The cost printed by bioptim mixes the two terms (36.0, 55.2, 133.2, 870.4) and is not shown in the video.

## Exercises
1. Repeat with `ControlType.CONSTANT` and check that the solution does not change; then switch to
   `TorqueBiorbdModel` -> `TorqueDerivativeBiorbdModel` and penalise `key="taudot"`.
2. Sweep w from 0.01 to 1000 and plot peak |dtau/dt| against the final-time torque and the integral of tau^2.
3. Penalise the derivative of `qdot` instead (`key="qdot"`) and compare with penalising `tau`.
