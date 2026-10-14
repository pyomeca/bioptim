# Panorama of the penalty library (anim_panorama.py, PenaltyPanorama, about 14 s)

## What it teaches
1. Beat 1: the penalties of `ObjectiveFcn.Lagrange` (37 names), `ObjectiveFcn.Mayer` (35) and `ConstraintFcn` (40) grouped
   by what they act on (controls, states, time, markers, segments, center of mass, contacts/forces, power/energy,
   continuity, stochastic, other), with per-group counts. Names in the enums can be aliases of one function
   (28 / 29 / 40 distinct functions), e.g. `TRACK_CONTROL` is `MINIMIZE_CONTROL`.
2. Beat 2: how to pick. Lagrange = integral over the intervals (`Node.ALL_SHOOTING`, the default; another node raises
   a `RuntimeError`), Mayer = one node (`Node.END` by default), constraint = `min_bound <= g <= max_bound`
   (both 0 by default, i.e. `g = 0`). Objectives are minimized, constraints must hold.

## Real data
Everything comes from the enums of this version, no solver: `generate_panorama_data.py` reads `__members__` (which
includes aliases), groups the names with regular expressions (first match wins, see `GROUPS`) and stores names, groups
and canonical names in `data/panorama_library.npz`. The scene recomputes the counts from the npz and asserts that every
displayed example name exists in its enum and that the groups cover each name exactly once. The generator also builds
the three code lines shown in beat 2 with `ObjectiveList.add` / `ConstraintList.add` (it raises on a wrong name or
keyword).

## Commands
    E=/c/Users/micka/miniconda3/envs/captury_biobuddy; export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
    PYTHONPATH=. python docs/animations/generate_panorama_data.py       # from the repo root
    cd docs/animations && manim render -qh anim_panorama.py PenaltyPanorama

## Caveats
- The grouping is my own regex classification of the names, not something bioptim defines; a name is in one group only
  (e.g. `TRACK_MARKER_WITH_SEGMENT_AXIS` is counted in markers, `QDDOT` in controls, `TORQUE_MAX_FROM_Q_AND_QDOT` in
  power/energy, `TRACK_PARAMETER` and `CUSTOM` in other).
- Only two example names per card are displayed; the counts cover all names. The multinode, parameter and phase
  transition families (`MultinodeConstraintFcn`, `ObjectiveFcn.Parameter`, ...) are not part of the panorama.
- The Constraint default node is not shown as a claim: the constraint example gives `node=Node.END` explicitly.

## Exercises
1. Print the `OTHER` group for each family and decide whether a new group is needed; how do the counts change?
2. Find two names of `ConstraintFcn` that have no `ObjectiveFcn` counterpart, and one objective without a constraint.
3. Add a Lagrange `MINIMIZE_CONTROL` and a Mayer `MINIMIZE_TIME` to a pendulum problem, and try
   `Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.END, key="tau")` to trigger the RuntimeError.
