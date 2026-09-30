# Tracking a reference: TRACK_STATE (scene `TrackState`, 14.5 s)

## What it teaches
The cart-pole `pendulum.bioMod` (actuated cart translation, passive pole rotation `q[1]`, 40 shooting nodes,
T = 2 s, RK4 with 5 steps) must make the pole angle follow the reference theta_ref(t) = 0.3 sin(2 pi t / T).
The tracking term is
`Objective(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=[1], node=Node.ALL, target=q_ref, weight=w)`
(`q_ref` has shape `(1, n_shooting + 1)`), balanced against `MINIMIZE_CONTROL` on `tau` (weight 1). Three real solves
with w = 1, 30, 1000, plotted as theta vs reference, the error e = theta - theta_ref (own axis) and tau(t):

| weight | rms e (rad) | max abs e (rad) | effort int tau^2 dt | IPOPT status / iterations |
|---|---|---|---|---|
| 1 | 0.221 | 0.399 | 0.0024 | 0 / 6 |
| 30 | 0.166 | 0.288 | 0.58 | 0 / 6 |
| 1000 | 0.0176 | 0.0252 | 7.41 | 0 / 8 |
| hard `ConstraintFcn.TRACK_STATE` | 1e-31 | 1e-30 | 16.5 | 0 / 5 |

Second beat: the hard-constraint alternative `constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=[1],
node=Node.ALL, target=q_ref)` forces e = 0 at every node.

## Commands
From the repo root (conda env `captury_biobuddy` on the PATH): `PYTHONPATH=. python docs/animations/generate_track_data.py`
(writes `data/track_pendulum.npz`), then from `docs/animations`: `manim render -qh anim_track.py TrackState`.

## Honest caveats
- Effort is computed as sum(tau_k^2) * dt on the piecewise-constant torque (the cart force), not read from the IPOPT cost.
- Only node values of theta are compared to the reference (error at the 41 nodes, not between them).
- The cart is free to travel: it reaches -3.0 m (weight 1000) and -3.2 m (hard). I widened the cart range from
  [-1, 5] to [-4, 4] in the script; with the default -1 bound the tracking was limited by the bound (and the hard version
  ended with IPOPT status 1). The cart travel is a real property of tracking a tilted hanging pole; nothing penalizes it.
- The hard-constraint torque chatters between nodes (about +-5 N zig-zag): exact tracking of a second-order system with
  a staircase control is an ill-conditioned inversion, and only `MINIMIZE_CONTROL` regularizes it. The soft solution at
  weight 1000 is smooth. This is a numerical feature of the solution, not a bug in the video.
- Weights 1 and 30 barely track (a small weight on a sine of 0.3 rad); values 0.1 and 10 gave almost no tracking at all
  with this problem scaling, hence 1/30/1000. The runs are warm started (10 -> 1000 -> hard; weight 1 restarts from
  weight 30); cold starts at 30, 1000 reach the same values (checked).
- All four solves converged (status 0); the problems are small and no poor local minima were observed, but only one
  reference and one discretization were tried.

## Exercises
1. Sweep the weight between 1 and 10^4 and plot rms error against effort: where is the knee of the trade-off curve?
2. Add `ObjectiveFcn.Lagrange.MINIMIZE_STATE` on `q[0]` (cart position) to stop the cart drifting: what does it cost in tracking?
3. Track `qdot` as well (`key="qdot"`, target = derivative of the reference): does the chattering of the hard version disappear?
