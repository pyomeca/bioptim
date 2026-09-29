# Holonomic constraints (anim_holonomic.py)

Teaches: a closed-loop condition (marker_1 of segment 0 == marker_3 of segment 1) turns two single pendulums into a
double pendulum. q = (theta0, y1, z1, theta1) is partitioned into independent u = (theta0, theta1) (states and
controls of the OCP: `q_u`, `qdot_u`, `tau`) and dependent v = (y1, z1) recovered from the constraint.
Code shown: `HolonomicConstraintsList().add(..., HolonomicConstraintsFcn.superimpose_markers, ...)`,
`HolonomicTorqueBiorbdModel(..., independent_joint_index=[0, 3], dependent_joint_index=[1, 2])`,
`compute_all_states_from_u_iterative` (all names checked in bioptim/examples/toy_examples/holonomic_constraints/two_pendulums.py).

Generate / render (repo root, then docs/animations):
    PYTHONPATH=. python docs/animations/generate_holonomic_data.py      # ~15 s, writes data/holonomic_two_pendulums.npz
    manim render -qh anim_holonomic.py HolonomicDoublePendulum          # ~15 s video

Caveats
- Data is the real IPOPT solution of the example (N = 30, MINIMIZE_TIME bounded to [0.5, 0.6], status optimal, 9 iterations;
  the time hits its 0.6 s upper bound).
- The residual is ~1e-16 by construction at the nodes: v is computed from u by solving the constraint
  (compute_all_states_from_u_iterative), so it measures the projection accuracy, not an IPOPT constraint violation.
  Between nodes (inside RK4 steps) it was not evaluated.
- The video is slowed down 10x (0.6 s of motion in 6 s). The example's own main() fails at `tau[:, :-1] = controls["tau"]`
  (shape 2 vs nb_tau = 4); the generator writes the two actuated torques into rows [0, 3].
- The plane drawn is (y, z) of the model frame, z up; segment 1 origin = dependent joint.

Exercises
1. Change `index=slice(1, 3)` to `slice(1, 2)` (only y) and see what the model does / what becomes under-constrained.
2. Make joint 3 (theta1) dependent instead (see two_pendulums_algebraic.py) and compare the OCP variables.
3. Lower n_shooting to 10 and check the residual and the cost.
