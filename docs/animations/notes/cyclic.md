# Cyclic NMPC (`CyclicNonlinearModelPredictiveControl`)

Scene `CyclicNMPC` in `anim_cyclic.py` (about 13 s, one scene). Data: `data/cyclic_results.npz`, made by
`generate_cyclic_data.py`.

## What it teaches

- In a cyclic NMPC the window IS one cycle (`cycle_len` nodes, `cycle_duration` seconds) and the window advances by a
  whole cycle at each solve (plain NMPC advances by one node). `update_function(nmpc, cycle, sol)` is called before
  each solve; return False to stop. Here it changes the target amplitude of the cart reference (0.4, 0.4, 0.7, 0.7 m).
- The "cyclic constraint" is implemented with bounds (`CyclicRecedingHorizonOptimization` in
  `bioptim/optimization/receding_horizon_optimization.py`): after each solve, `advance_window_bounds_states` sets the
  initial bound to the last state and `_set_cyclic_bound` bounds the last node to that same state +/- 1 % of
  `x_max - x_min`. The initial guess is the previous solution. `x_bounds` and `u_bounds` must be of type
  `CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT` (what `model.bounds_from_ranges` returns).
- Concatenated result: 4 cycles x 4 s (20 nodes of 0.2 s each), cart-pendulum from the MHE examples, one IPOPT solve
  per cycle.

## Commands (from the repo root, env captury_biobuddy)

```bash
PYTHONPATH=. python docs/animations/generate_cyclic_data.py
cd docs/animations && manim render -qh anim_cyclic.py CyclicNMPC
```

## Results (all real; computed from the npz)

- 4 solves, IPOPT status 0 for all, 7 / 7 / 11 / 7 iterations.
- max |x_last - x_first| over the 4 states (cart q, theta, cart qdot, theta_dot): 0.017, 0.031, 0.120, 0.120.
- Cycle N+1 starts exactly at the last node of cycle N (the concatenated trajectory has no jump).

## Honest caveats

- The cyclic constraint is NOT an equality. It is a +/- 1 % of the range box, and from the second solve on it is
  centred on the end of the previous cycle (not on the start of the first). The first solve has no anchor: its last node
  is only bounded by the whole range (the reference is periodic, so it comes out nearly periodic anyway, gap 0.017).
- Cycles 3 and 4 (amplitude change) sit exactly at the edge of the slack on both velocities (0.12 = 1 % of the
  velocity range 12 rad/s or m/s): the state drifts by the maximum allowed per cycle. This is the cost of changing
  the task while keeping the cyclic bound.
- The default model ranges (cart +/-5 m, velocities +/-10 pi) give a much looser slack (0.1 m, 0.63); I narrowed
  `x_bounds` in the script (q in [-1, 1] x [-pi, pi], qdot in [-6, 6]) to get a tighter cycle. This is visible in
  the code of `generate_cyclic_data.py` but not in the video.
- A period of 2 s (near the pendulum resonance) gave poor local minima (the cart did not follow the sine after
  cycle 1); 3 s was mediocre; 4 s tracks well, so the 4 s period was kept.
- The example `cyclic_nmpc.py` (arm2, wheel rotation) is quasi-cyclic: it overrides
  `advance_window_bounds_states` to reset the wheel angle to -pi.

## Exercises

1. Change the slack: replace the 0.01 in `_set_cyclic_bound` (subclass and override it) by 0.001 and re-run. Do
   cycles 3 and 4 still converge? What happens to the tracking of the new amplitude?
2. Set `AMPS = [0.4] * 4` in the generator: the four cycles should be almost identical. Compute the maximal
   difference between cycle 2 and cycle 4.
3. Use `MultiCyclicNonlinearModelPredictiveControl` (see `multi_cyclic_nmpc.py`) with `n_cycles_simultaneous=2` and
   compare the number of IPOPT iterations per solve.
