"""
This example aims to replicate the example provided in Rockit: matrix_lyapunov.py
It uses the Lyapunov differential equation to approximate state covariance along the trajectory.
"""

import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
import pickle

from bioptim import (
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    OptimalControlProgram,
    NonLinearProgram,
    DynamicsList,
    BoundsList,
    InterpolationType,
    SocpType,
    Node,
    ConstraintList,
    ConstraintFcn,
    InitialGuessList,
    PenaltyController,
    ConfigureProblem,
    OdeSolver,
    StochasticBioModel,
    PhaseDynamics,
)
from bioptim.examples.stochastic_optimal_control.rockit_model import RockitModel
from bioptim.examples.stochastic_optimal_control.common import (
    test_matrix_semi_definite_positiveness,
    test_eigen_values,
    reshape_to_matrix,
)


def configure_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, algebraic_states, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, algebraic_states, nlp, with_noise=False
        ),
    )


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    # Algebraic states variables
    ConfigureProblem.configure_stochastic_m(
        ocp, nlp, n_noised_states=2, n_collocation_points=nlp.model.polynomial_degree + 1
    )
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=2)
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, algebraic_states, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, algebraic_states, nlp, with_noise=False
        ),
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, algebraic_states, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, algebraic_states, nlp, with_noise=True
        ),
        allow_free_variables=True,
    )


def cost(controller: PenaltyController):
    q = controller.states["q"].cx
    return (q - 3) ** 2


def bound(t):
    return 2 + 0.1 * cas.cos(10 * t)


def path_constraint(controller, is_robustified: bool = False):
    t = controller.time.cx
    q = controller.states["q"].cx
    sup = bound(t)
    if is_robustified:
        P = StochasticBioModel.reshape_to_matrix(
            controller.algebraic_states["cov"].cx, controller.model.matrix_shape_cov
        )
        sigma = cas.sqrt(cas.horzcat(1, 0) @ P @ cas.vertcat(1, 0))
        sup -= sigma
    return q - sup


def prepare_socp(
    final_time: float,
    n_shooting: int,
    motor_noise_magnitude: np.ndarray,
    polynomial_degree: int,
    is_stochastic: bool = False,
    is_robustified: bool = False,
    socp_type: SocpType = SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
    expand_dynamics: bool = True,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> StochasticOptimalControlProgram | OptimalControlProgram:
    problem_type = socp_type

    bio_model = RockitModel(
        socp_type=problem_type,
        motor_noise_magnitude=motor_noise_magnitude,
        polynomial_degree=polynomial_degree,
    )

    nb_q = bio_model.nb_q
    nb_u = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(cost, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL, quadratic=False)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=0, node=Node.START, target=0.5)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", index=0, node=Node.START, target=0)
    constraints.add(
        path_constraint,
        is_robustified=is_robustified,
        min_bound=-cas.inf,
        max_bound=0,
        node=Node.ALL,
    )

    x_bounds = BoundsList()
    x_bounds["q"] = [-0.25] * nb_q, [cas.inf] * nb_q

    u_bounds = BoundsList()
    u_bounds["u"] = [-40] * nb_u, [40] * nb_u

    # Dynamics
    dynamics = DynamicsList()

    if is_stochastic:
        dynamics.add(
            configure_stochastic_optimal_control_problem,
            dynamic_function=lambda time, states, controls, parameters, algebraic_states, nlp, with_noise: bio_model.dynamics(
                states,
                controls,
                parameters,
                algebraic_states,
                nlp,
                with_noise=with_noise,
            ),
            phase_dynamics=phase_dynamics,
            expand_dynamics=expand_dynamics,
        )

        a_init = InitialGuessList()
        a_init.add(
            "m",
            initial_guess=[0] * bio_model.matrix_shape_m[0] * bio_model.matrix_shape_m[1],
            interpolation=InterpolationType.CONSTANT,
        )

        cov0 = np.diag([0.01**2, 0.1**2]).reshape((-1,), order="F")
        a_init.add(
            "cov",
            initial_guess=cov0,
            interpolation=InterpolationType.CONSTANT,
        )
        constraints.add(ConstraintFcn.TRACK_ALGEBRAIC_STATE, key="cov", node=Node.START, target=cov0)
        constraints.add(
            ConstraintFcn.TRACK_ALGEBRAIC_STATE, key="cov", node=Node.ALL, min_bound=1e-6, max_bound=cas.inf
        )

        return StochasticOptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            a_init=a_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            n_threads=6,
            problem_type=problem_type,
        )

    else:
        dynamics.add(
            configure_optimal_control_problem,
            dynamic_function=lambda time, states, controls, parameters, algebraic_states, nlp, with_noise: bio_model.dynamics(
                states,
                controls,
                parameters,
                algebraic_states,
                nlp,
                with_noise=with_noise,
            ),
            phase_dynamics=phase_dynamics,
            expand_dynamics=expand_dynamics,
        )

        ode_solver = OdeSolver.COLLOCATION(polynomial_degree=socp_type.polynomial_degree, method=socp_type.method)

        return OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            ode_solver=ode_solver,
            n_threads=6,
        )


def main():
    """
    Prepare, solve and plot the solution
    """
    is_stochastic = True
    is_robust = True
    if not is_stochastic:
        is_robust = False

    polynomial_degree = 5

    # --- Prepare the ocp --- #
    socp_type = SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre")
    n_shooting = 40
    final_time = 1
    dt = final_time / n_shooting
    ts = np.arange(n_shooting + 1) * dt
    motor_noise_magnitude = np.array([0])

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")

    socp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=motor_noise_magnitude,
        is_stochastic=is_stochastic,
        is_robustified=is_robust,
        socp_type=socp_type,
    )

    sol_socp = socp.solve(solver)
    T = sol_socp.time
    q = sol_socp.states["q"]
    u = sol_socp.controls["u"]

    # sol_ocp.graphs()
    plt.figure()
    plt.plot(T, bound(T), "k", label="bound")
    plt.plot(np.squeeze(T), np.squeeze(q), label="q")
    plt.step(ts, np.squeeze(u / 40), label="u/40")

    if is_stochastic:
        cov = sol_socp.decision_algebraic_states["cov"]

        o = np.array([[1, 0]])
        sigma = np.zeros((1, n_shooting + 1))
        for i in range(n_shooting + 1):
            cov_i = cov[:, i]
            if not test_matrix_semi_definite_positiveness(cov_i):
                print(f"Something went wrong at the {i}th node. (Semi-definiteness)")

            if not test_eigen_values(cov_i):
                print(f"Something went wrong at the {i}th node. (Eigen values)")

            P = reshape_to_matrix(cov_i, (2, 2))
            sigma[:, i] = np.sqrt(o @ P @ o.T)

        plt.plot(
            [ts, ts],
            np.squeeze([q[:, :: polynomial_degree + 2] - sigma, q[:, :: polynomial_degree + 2] + sigma]),
            "k",
        )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
