# Visualising a solution with sol.animate (anim_viz.py, scene Visualization)

## What the scene teaches
`sol.animate(viewer=..., show_now=..., n_frames=...)` opens the solution in a viewer, and the two viewers do not treat
the frames the same way. One real solve, then the viewers are really called:

* Problem: `double_pendulum.bioMod` (torque driven, planar), N = 30 intervals, T = 1 s, RK4 (5 steps), minimise the
  squared torque (Lagrange), q(0) = (0, 0), q(T) = (1.5, -1.0) rad, rest to rest, |tau| <= 30 N.m.
  IPOPT: status 0 (Solve_Succeeded), 15 iterations, cost 227.08, max |tau| = 17.6 N.m.
* pyorerun (installed here: pyorerun 1.2.3, rerun-sdk 0.21.0) is REALLY run, without a window:
  `sol.animate(viewer="pyorerun", show_now=False, n_frames=200)` logs into rerun's in-memory recording, `rr.save(...)`
  writes it to an `.rrd` file (95 kB, 6 entity paths, 31 time stamps). The file is read back with
  `rerun.dataframe` and the four marker positions pyorerun logged at every time stamp are what the video re-draws as a
  stick figure (max gap with the biorbd markers computed independently from q: 5.6e-8 m, float32 round-off).
* `n_frames` is ignored by the pyorerun viewer in this version (`Solution.animate` passes it to
  `BiorbdModel.animate`, which calls `animate_with_pyorerun(ocp, solution, show_now, show_tracked_markers, **kwargs)`
  without it): 31 time stamps with `n_frames=200` and with the default `n_frames=0`. pyorerun plays `decision_states`
  at the nodes (`SolutionMerge.NODES`) with their times.
* bioviz is NOT installed here. `sol.animate(viewer="bioviz", ...)` really raises
  `RuntimeError: bioviz must be install to animate the model` (shown in the video). What the bioviz path would receive is
  computed with the library's own `interpolate_data` (`viewer_bioviz.py`): 31 frames for `n_frames=0`, 200 for
  `n_frames=200` (evenly spaced in time). bioviz itself was never run.

The video shows the exact code that ran (simplified: `Solver.IPOPT()` options and the `**kwargs` of the helper are not
shown), the stick figure, the frame ticks of each viewer and the error.

## Commands (repo root, env captury_biobuddy; render with the manim venv)
    E=/c/Users/<you>/miniconda3/envs/captury_biobuddy
    export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"; export PYTHONIOENCODING=utf-8
    PYTHONPATH=. python docs/animations/generate_viz_data.py
    python docs/animations/render_series.py anim_viz.py Visualization --lang both --strict --quality 1080p30

The generator prints: `status 0 iterations 15 cost 227.07996951669958`, `pyorerun n_frames=200 -> time stamps 31
default n_frames=0 -> 31`, `entities in the recording: 6  file size (bytes): 95202 95201`,
`viewer='bioviz' -> RuntimeError: bioviz must be install to animate the model`,
`frames handed to bioviz: n_frames=0 -> 31  n_frames=200 -> 200`. The `.rrd` files are written to a temporary folder,
never to the repository.

## Honest caveats
- NOT a screen capture of any viewer: a GUI window cannot be captured reliably, and none was. The stick figure is
  re-drawn by Manim from the positions read back from the `.rrd` (a real pyorerun output), and labelled as such. Rerun's
  own viewer (`rerun file.rrd`) was not opened; rerun 0.21 has no headless screenshot or video export, so no real image
  of the viewer is included.
- The figure steps through the 31 nodes (latest frame logged), as rerun shows a time-stamped recording; it is not
  interpolated. The playback speed in the video is not real time.
- bioviz: not installed, not run. `Solution.animate` requires bioviz >= 2.5.0 and < 2.6.0 (`check_version` in
  `viewer_bioviz.py`) and, in the bioviz path, `show_now=True` blocks until the window is closed. The bioviz frame counts
  come from `interpolate_data`, they are not observed in a viewer. In this version `sol.interpolate` is a linear
  interpolation of the stepwise states, so the 200 frames are not new information.
- Defaults differ: `Solution.animate(viewer="bioviz")` but `BiorbdModel.animate(viewer="pyorerun")`; since `Solution.animate`
  always passes `viewer=`, the effective default of `sol.animate()` is bioviz. `basic_ocp.py` keeps the call commented.
- With `show_now=False` the pyorerun path calls `prerun.rerun(notebook=True)`: no viewer is spawned, data stay in the
  in-memory recording until a sink is connected (here `rr.save`). With `show_now=True` a rerun viewer window is spawned.
- pyorerun 1.2.3 pins `numpy==1.26.4` and `rerun-sdk==0.21.0` in its metadata; the environment used here has numpy 2.4.6
  (it ran, but it is not the pinned combination).
- Single small solve, one local minimum found (non-convex problem); the motion is not the point.

## Exercises
1. Call `sol.animate(viewer="pyorerun", show_now=False)` with `n_frames=0` and with `n_frames=500`, save both with
   `rr.save` and count the time stamps with `rerun.dataframe`: are they equal?
2. Install bioviz (version 2.5.x) and run `sol.animate(viewer="bioviz", n_frames=0)` then `n_frames=200`: how many frames
   does the time slider have? Compare with `interpolate_data`.
3. Make the solve two phases (see `SolutionTour`) and compare, for bioviz, `n_frames=0` (phases merged if possible) with
   `n_frames=-1` (not merged) as the docstring of `Solution.animate` says: how many windows open?
