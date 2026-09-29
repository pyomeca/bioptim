# Tracking measured markers: TRACK_MARKERS (scene `TrackMarkers`, ~13.6 s)

## What it teaches
`ObjectiveFcn.Lagrange.TRACK_MARKERS` with `target=` a measured marker trajectory makes the model markers follow the
data (the basis of marker-based inverse kinematics / motion tracking). The model markers are shown converging onto the
measurements over IPOPT iterations (0, 1, 2, 4, 6, 8, 10, 14), then the final solution is played back in time, with the
per-node error on its own log axis and the noise level as a reference line.

Data are SYNTHETIC: a known joint motion q_true(t) (q1 = sin(2 pi t / T), q2 = 0.6 (1 - cos(2 pi t / T))) of
`bioptim/examples/models/double_pendulum.bioMod` (planar, y-z plane) is passed through biorbd's marker function, then
Gaussian noise (sigma = 1 cm per axis, seed 0) is added. Tracked: `marker_2` (elbow) and `marker_4` (tip),
`marker_index=[1, 3]`, `axes=[Axis.Y, Axis.Z]`, target shape (2 axes, 2 markers, N+1). N = 30, T = 1.5 s, RK4 (5 steps),
free initial/final states, `TorqueBiorbdModel`, `MINIMIZE_CONTROL` on tau with weight 1e-3 as regulariser.

## Real numbers (computed from `data/markers_tracking.npz`)
- IPOPT: status 0 (Solve_Succeeded), 14 iterations, cost 3.248.
- RMS marker error vs measured: 113 cm (iteration 0, q = 0) -> 11 cm (it. 8) -> 1.4 cm (it. 10) -> 1.31 cm (converged).
- Noise RMS (measured vs noise-free markers): 1.59 cm. Final model vs noise-free truth: 0.84 cm RMS, i.e. tracking
  filters part of the noise. RMS joint-angle error vs q_true: 0.64 deg.

## Commands
Repo root, conda env `captury_biobuddy` on the PATH: `PYTHONPATH=. python docs/animations/generate_markers_data.py`
(writes `data/markers_tracking.npz`). Then from `docs/animations`: `manim render -qh anim_markers.py TrackMarkers`.

## Honest caveats
- "Iteration k" is a separate solve with `set_maximum_iterations(k)` (deterministic IPOPT), so it is the k-th iterate;
  the intermediate ones report status 1 (max iterations), only the last one converged. The iteration-0 trajectory is the
  default initial guess (q = 0), not a good guess.
- Only the y-z coordinates are tracked and the motion is planar; noise is added on y and z only. The x column of the
  target is not used. No torque bound: the recovered torques are whatever tracking requires (not analysed here).
- The synthetic motion is prescribed in q (not obtained from a forward simulation), so tau_true is unknown and
  the fit is purely kinematic. One noise seed only.
- The error axis is logarithmic so that 113 cm and 1 cm fit on one plot; the weight 1000 is not tuned.

## Exercises
1. Raise sigma to 3 cm and 5 cm: how do the final "vs measured" and "vs truth" errors evolve? Does the fit still filter noise?
2. Track only the tip marker (`marker_index=[3]`): is the elbow angle still recovered? Compare RMS q error.
3. Lower the weight (1000 -> 1) or raise the tau weight: when does the regulariser start to dominate the tracking?
