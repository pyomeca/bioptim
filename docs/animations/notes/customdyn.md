# Custom dynamics: your own model (anim_customdyn.py)

## What the scene teaches
A model that does not come from biorbd plugs into a Bioptim OCP as a subclass of `StateDynamics` (`bioptim/dynamics/state_space_dynamics/abstract_dynamics.py`).
The user provides: `name` and `name_dofs`, `state_configuration_functions` (here a hand-made variable `q` through
`ConfigureVariables.configure_new_variable(..., as_states=True)` and `States.QDOT`), `control_configuration_functions` (`Controls.TAU`),
`algebraic_configuration_functions` and `extra_configuration_functions` (both empty), and `dynamics(...)`, which returns
`DynamicsEvaluation(dxdt=..., defects=None)`, the state derivative. The variable names declared there ("q", "qdot", "tau") are the
keys used by `x_bounds`, `u_bounds`, initial guesses and objectives (`ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"`).

The scene shows the pieces of `DampedPendulum` (class of `generate_customdyn_data.py`, adapted from
`bioptim/examples/toy_examples/custom_model/custom_package/my_model.py`), the state derivative as text, then two real solves of the
OCP `prepare_ocp` of `bioptim/examples/toy_examples/custom_model/main.py` (swing from q = 0 to q = pi rad, N = 30, T = 1 s, RK4 with 5 steps,
minimise the integral of tau^2, |tau| <= 20 N.m) that differ only by the viscous friction term `-damping * qdot`
(damping d = 0 as in the shipped example, and d = 1 N.m.s/rad).

Numbers shown (generator output):
- free (d = 0): IPOPT status 0, 35 iterations, cost 134.0013; the example's own `MyModel` gives the same cost (134.0013, 35 iterations).
- damped (d = 1): IPOPT status 0, 106 iterations, cost 108.5161.
- energy released by gravity between q = 0 and q = pi: 18.721 J; energy dissipated by the friction, integral of d qdot^2 dt: 11.984 J;
  work of the torque 6.759 J absorbed (energy balance residual -0.022 J, discretisation).
- max |tau_damped - tau_free| = 4.894 N.m; max |q_damped - q_free| = 0.0044 rad (the motion is almost identical, only the torque changes).

## Commands
    E=/c/Users/<you>/miniconda3/envs/captury_biobuddy
    export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python docs/animations/generate_customdyn_data.py    # -> data/customdyn_pendulum.npz (~10 s)
    python docs/animations/render_series.py anim_customdyn.py CustomDynamics --lang both --strict
    black -t py311 -l120 docs/animations/anim_customdyn.py docs/animations/generate_customdyn_data.py

## Honest caveats
- Code panel: simplified lines of `generate_customdyn_data.py` (`...`, the two `@property` lines of `name_dofs` folded into one, the long
  `forward_dynamics` expression broken over two lines, `defects=None` and `Solver.IPOPT(show_online_optim=False)` defaults omitted);
  `prepare_ocp(model=model, final_time=..., n_shooting=...)` is the function of the linked example, which is the reference.
- Sign convention of the shipped model: with `L = com[2] = -0.9542` the torque-free equation is `qddot = 9.36 sin(q) / 0.95`, so q = 0 is the
  UNSTABLE equilibrium and q = pi the stable one. The example's comments call it a swing-up, but the motion computed here falls from q = 0
  to q = pi; the work of the torque is negative (it brakes). That is why friction lowers the cost: it does part of the braking.
- Single start (the example's initial guess: q and qdot initial guess 20, tau 10), no continuation, no warm start between the two solves.
  A non-convex problem returns a local minimum; other initial guesses were not tried. Both solves converged (status 0).
- The dissipated energy uses the trapezoid rule on the 31 node values of qdot; the work of the torque uses tau piecewise constant
  (tau dq per interval). The small balance residual comes from the integration scheme (RK4 with 5 steps inside each interval).
- The friction coefficient d = 1 N.m.s/rad is chosen for a visible effect, not identified on real data. Mass, inertia and centre of mass
  are the constants of the example. The model has one degree of freedom and no biorbd model file.
- `StateDynamics` is used directly, as the example does; the `BioModel` protocol (`bioptim/models/protocols/biomodel.py`) is a larger
  interface (markers, segments, ...) that this minimal model does not implement, and the OCP accepted it for a torque-driven problem.
- The README section "Custom dynamical model" writes `class MyModel(BioModel, AbstractModel)`; in this version the base class is `StateDynamics`
  (there is no `AbstractModel`), as in the example.

## Exercises
1. Set `damping = 3.0` in `DampedPendulum(damping=...)`: how do the cost, the dissipated energy and the iteration count change?
2. Add a second control-independent term to `forward_dynamics`, for example a constant Coulomb-like term `-0.2 * sign_smooth(qdot)` (use a smooth
   function such as `tanh(qdot / 0.1)`), and check that IPOPT still converges.
3. Rename the state `"q"` to `"theta"` in `state_configuration_functions` and fix `x_bounds` in `prepare_ocp` accordingly (the keys must match your names).
