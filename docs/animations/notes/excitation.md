# Excitation -> activation dynamics (scene `ExcitationActivation`, ~15 s)

## What it teaches
With `MusclesWithExcitationsBiorbdModel` the muscle excitation e(t) becomes the CONTROL (`u_bounds["muscles"]`) and the
activation a(t) becomes a STATE (`x_bounds["muscles"]`, hence a is continuous and bounded in [0, 1]). biorbd's
`activationDot` (exposed as `bio_model.muscle_activation_dot()`) gives da/dt = f(e, a): the activation lags the excitation.
The scene shows, for BICshort of the arm26 reaching problem (static_arm.py task, 30 nodes, T = 0.5 s, RK4):
e(t) staircase vs a(t), a zoom on one rise and one fall with the lag measured on the data (18 ms up, 14 ms down to
a = 0.5), and the resulting shoulder/elbow angles. Code lines: `MusclesWithExcitationsBiorbdModel(path,
with_residual_torque=True)`, the two bounds, `MINIMIZE_CONTROL key="muscles"`. Contrast (text only): with
`MusclesBiorbdModel` (anim_muscle.py) a is the control, so it can jump between nodes.

## Verified constants
The bioMod has no explicit time constants; I checked numerically that `muscle_activation_dot` equals the De Groote (2016)
formula da/dt = [ (f+0.5)/(tau_act (0.5+1.5a)) + (-f+0.5)(0.5+1.5a)/tau_deact ] (e-a), f = 0.5 tanh(0.1 (e-a)), with
tau_act = 0.01 s, tau_deact = 0.04 s (5 (e, a) test pairs, agreement to 1e-12). The constants live in biorbd (C++), not in bioptim.

## Commands
Repo root (env `captury_biobuddy`): `PYTHONPATH=. python docs/animations/generate_excitation_data.py`
(writes `data/excitation_arm.npz`, about 30-50 s), then from `docs/animations`: `manim render -qh anim_excitation.py ExcitationActivation`.

## Honest caveats
- IPOPT converged (status 0, 39 iterations, cost 2.01) but the hand ends 3.0 cm from the target (weight 1000, same as
  anim_muscle.py which gets 3.0 cm too); the joint motion is nearly identical to that activation-as-control solution. One local solution only.
- The "lag" is measured on the dense activation curve: the same casadi function integrated with RK4 (40 substeps per
  interval, excitation constant per interval). It matches the solution's node activations to 1.6e-4.
- The dynamics is first order but with a state-dependent rate (not a pure exponential); tau_act/tau_deact are nominal
  values. The "0.5-crossing" lag depends on the level reached by e (here 0.67 then 1 for the rise, 0 for the fall).
- The optimal excitation is nearly bang-bang and the initial activation is fixed to 0.1 (`x_bounds["muscles"][:, 0]`).
  The nodes (white dots) are only 16.7 ms apart, about the same as the time constants.

## Exercises
1. Add `ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", derivative=True` (smooth excitation): how do e(t) and the lag change?
2. Fix the initial activation to 0 or 1: does the first-node behaviour change, and what happens to the cost?
3. Track a reference excitation with `TRACK_CONTROL` as in `muscle_excitations_tracker.py`.
