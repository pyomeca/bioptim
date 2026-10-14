# Fatigue model on the torque (scene `FatigueXia`, ~10.7 s)

## What it teaches
`pendulum_with_fatigue.prepare_ocp(fatigue_type="xia", split_controls=False)` (pendulum.bioMod, 30 nodes, T = 1 s,
`MINIMIZE_CONTROL` on tau, |tau| <= 100 on the sliding joint) adds, per joint, an `XiaTauFatigue` made of two
`XiaFatigue` (tau_min and tau_max sides). Each side brings extra states: `tau_minus_ma/mr/mf` and
`tau_plus_ma/mr/mf` (active / resting / fatigued fractions, ma + mr + mf = 1). The scene plots tau on the sliding joint,
the active and fatigued fractions of both sides (dof 0, the actuated one) and the rotation theta, with the code lines
(`FatigueList`, `XiaTauFatigue`, `XiaFatigue`, `TorqueBiorbdModel(..., fatigue=)`, `FatigueBounds`,
`FatigueInitialGuess`) copied from the example.

## Commands
Repo root, env `captury_biobuddy`: `PYTHONPATH=. python docs/animations/generate_fatigue_data.py [T]` (writes
`data/fatigue_xia.npz`), then from `docs/animations`: `manim render -qh anim_fatigue.py FatigueXia`.

## Honest caveats
- Real solve, IPOPT status 0, 243 iterations, cost 41.66. Numbers on screen are computed from the npz.
- Fatigue is small over 1 s with LD=LR=100: fatigued fraction peaks at 4.3 %, resting fraction >= 75 %. The
  fractions axis is therefore tight (0 to 0.22). Longer T or an effort-heavy task is needed for strong fatigue.
- tau is a staircase control; the last node has no control, so the plot repeats the last value. The torque
  shows a sharp -22 N peak and a -13.7 N value at the end (braking so that qdot(T) = 0), not a defect of the plot.
- The resting fraction is not plotted (stated as 1 - active - fatigued and read out as a minimum).

## Exercises
1. Run with T = 2 or 3 and lower LR: how do the fatigued fractions change?
2. Switch `fatigue_type` to "michaud" or "effort" and compare states.
3. Set `split_controls=True` and see how tau_minus / tau_plus controls appear.
