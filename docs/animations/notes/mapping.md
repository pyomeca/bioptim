# Mapping (BiMapping / BiMappingList)

**What it teaches.** A `BiMappingList` reduces the number of optimised variables. On a double pendulum
(`bioptim/examples/models/double_pendulum.bioMod`, 2 actuated joints) the same task (q = [1, 1] rad at rest at t = 3 s,
30 shooting nodes, RK4, `MINIMIZE_CONTROL` on tau) is solved twice with real IPOPT runs:
without mapping (2 torques per node, decision vector 185) and with
`mappings.add("tau", to_second=[0, 0], to_first=[0])` passed as `variable_mappings=mappings` (one shared torque,
decision vector 155 = 185 - 30). The two torque curves collapse onto one (tau1 - tau2 plotted: max 7.9 -> 0).
`u_bounds["tau"]` must then have the size of the reduced vector (1 row).

**Commands** (repo root, env captury_biobuddy on the PATH):

    PYTHONPATH=. python docs/animations/generate_mapping_data.py      # writes data/mapping_double_pendulum.npz
    cd docs/animations && manim render -qh anim_mapping.py Mapping    # 12.5 s, 1080p60

**Real numbers** (all computed from the npz): free IPOPT status 0, 17 it, cost 92.5; mapped status 0, 12 it,
cost 71.3. Physical torque effort sum(tau1^2 + tau2^2)*dt: 92.5 (free) vs 142.6 (mapped, +54 %).

**Caveats.**
- The IPOPT costs are not comparable: the objective is evaluated on the reduced `tau`, so the mapped cost counts the shared
  torque once (71.3 = 142.6 / 2). The scene shows both numbers.
- Problem is non-convex; both solutions are local minima found from the default initial guess.
- A first attempt (T = 1.5 s, |tau| <= 60) with a shared torque was locally infeasible (IPOPT "local infeasibility"):
  one shared torque cannot reach every pose in a short time. The scene uses T = 3 s, which converges.
- Only a "share" mapping is shown; `oppose_to_second=` gives mirrored (sign-flipped) torques, not rendered here.

**Exercises.**
1. Use `to_second=[0, 0], oppose_to_second=[1]` (check signs in `bioptim/misc/mapping.py`) so tau2 = -tau1 and find a target pose reachable that way.
2. Reduce `phase_time` to 1.5 s and observe when the mapped problem becomes infeasible; what does that say about actuator sharing?
3. Print `ocp.nlp[0].controls.shape` / `sol.vector.shape` for both problems and explain the 30-variable difference.
