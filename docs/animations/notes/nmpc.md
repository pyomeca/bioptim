# NMPC: receding horizon (anim_nmpc.py, scene NMPCWindow, about 15 s)

## What it teaches
A window of 1 s (10 nodes) is solved, only the first control (and the resulting next state) is applied, then the window
slides by ONE node, the reference inside the window is updated (`update_objectives_target`) and the OCP is solved again,
warm-started from the previous solution shifted by one node. Faded orange = prediction over each window, bold green =
applied (first) node of each window, dashed = cyclic reference (sine, amplitude 0.5 m, period 2 s).

## Real content
`generate_nmpc_data.py` runs `NonlinearModelPredictiveControl` (`get_all_iterations=True`) on
`examples/models/cart_pendulum.bioMod` (same model as `toy_examples/moving_horizon_estimation/mhe.py`): 30 IPOPT solves,
all status 0 (6-9 iterations each), stored in `data/nmpc_results.npz` (predicted q / tau of every window + applied
trajectory). Code lines shown in the video are the ones used in the script (simplified names: `ref(step)`, `objectives`).

## Commands (from the repo root / docs/animations)
    PYTHONPATH=. python docs/animations/generate_nmpc_data.py        # conda env captury_biobuddy
    cd docs/animations && manim render -qh anim_nmpc.py NMPCWindow

## Caveats
- The plain (non-cyclic) `NonlinearModelPredictiveControl` is used instead of `cyclic_nmpc.py` (which advances a full
  cycle per solve, so there is no sliding window to show). The task is cyclic through the sine reference.
- There is no plant/model mismatch or noise: the "applied" state is simply node 1 of the previous window (it becomes the
  fixed initial state of the next window), i.e. nominal MPC. In closed loop with a real system the measured state would
  replace it.
- The initial 0.18 m tracking error and the slight amplitude lag come from the cart starting at rest with a
  1 s horizon, the 0.01 weight on the force and the 1.0 weight on qdot. Only the cart is actuated; the pendulum swings
  freely (|theta| < 1.4 rad).
- Not real time: solves took a few ms each here, but no timing claim is made in the video.

## Exercises
1. Shorten the window (`N = 5`, then 3) and watch the tracking degrade or the solver fail: why does a short horizon lag?
2. Raise the weight on `MINIMIZE_CONTROL` (0.01 -> 1) and compare the applied force with the reference tracking.
3. Add a disturbance: after each solve, perturb the applied next state (e.g. subtract 0.05 m from q) in
   `advance_window_bounds_states` and observe how the predictions correct it.
