# Discrete mechanics (variational integrator)

Scene `DiscreteMechanics` in `anim_variational.py` (about 16.5 s, one scene, two beats).

## What it teaches
1. With `VariationalOptimalControlProgram` the state is only `q_k` (no velocity state, no Runge-Kutta): the ODE is
   replaced by the discrete Euler-Lagrange equations, one per node triplet
   (`OdeSolver.VARIATIONAL()` is a placeholder, `skip_continuity=True`, the continuity is a three-node
   `multinode_constraint` calling `model.discrete_euler_lagrange_equations`).
2. Geometric property: on a free pendulum the total-energy error of the variational scheme stays bounded, whereas
   explicit schemes drift.
3. A real variational OCP (pendulum swing-up, IPOPT) with the exact code lines.

## What is real
`generate_variational_data.py` -> `data/variational_pendulum.npz`. Model: `models/variational_pendulum.bioMod`
(1 kg point mass, 1 m, q = 0 hanging down).
* Beat 1, free motion (tau = 0), Δt = 0.1 s for all schemes, 6000 steps (600 s), released at rest from 90 deg:
  * variational: the model's own `discrete_euler_lagrange_equations` (and `compute_initial_states` for the first
    step, with qdot0 = 0) solved for `q_{k+1}` with a casadi Newton root finder. The velocity is deduced from the
    discrete momentum `p_k = D2 Ld(q_{k-1}, q_k)` and `qdot_k = M(q_k)^-1 p_k`. This is done OUTSIDE an OCP (no
    IPOPT): it is the same residual as the OCP constraints, solved forward instead of as a constraint.
  * RK1 and RK4 are hand-written steps on `model.forward_dynamics` (not bioptim's `OdeSolver` classes).
  * Energy for all three: `E = L(q, qdot) - 2 L(q, 0)` with the model's own Lagrangian (T + V, V = 0 at the pivot
    height). The plot shows `E(t) - E(0)`.
* Beat 2: a real `VariationalOptimalControlProgram` (50 intervals, 2 s, |tau| <= 25, MINIMIZE_CONTROL,
  q(0) = 0, q(T) = pi, qdot(0) = qdot(T) = 0), IPOPT status 0, 17 iterations. Torque shown at the nodes (linear
  continuous control; the last node control is unused).

Numbers (from the npz): variational `E - E0` in [-0.20, 0.00] J over the 600 s; RK4 -0.42 J at 600 s, drifting
linearly (-0.14 J at 200 s), leaving the variational band at 284 s; RK1 +46 J at 10 s, +500 J at 600 s.

## Honest caveats
* With Δt = 0.1 s RK4 is MORE accurate than the variational scheme for the first 284 s (max error 0.2 J vs 0.02 J at
  30 s): the variational error is larger but does not grow. The message is boundedness, not accuracy at short times.
* The variational energy error oscillates within the band with the swing (the velocity is a post-processed
  quantity), the plot looks like a filled band at 600 s.
* The discrete scheme is the default `QuadratureRule.TRAPEZOIDAL` discrete Lagrangian; other rules give other
  errors. The RK1 curve leaves the axis after a few steps (annotated with its numbers); after a while the pendulum
  spins, so the 500 J is not a physical energy of a meaningful motion.
* Beat 1 is a forward simulation, not an optimization.

## Commands
```
E=/c/Users/micka/miniconda3/envs/captury_biobuddy; export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
PYTHONPATH=. python docs/animations/generate_variational_data.py
cd docs/animations && manim render -qh anim_variational.py DiscreteMechanics
```

## Exercises
1. Change `DT` to 0.02 and 0.2: how do the variational band and the RK4 drift scale (about Δt^2 and Δt^4)? At which Δt
   does RK4 become worse than the variational scheme within 60 s?
2. Add a small constant torque (tau = 0.5 N·m) in `free_motion` (control arguments of the DEL equations) and compare
   the energy balance with the work of the torque.
3. Change `discrete_approximation` of the model (`QuadratureRule.MIDPOINT`) and look at the error band.
