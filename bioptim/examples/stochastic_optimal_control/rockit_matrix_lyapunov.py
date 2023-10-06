"""
This example aims to replicate the example provided in Rockit: matrix_lyapunov.py
It uses the Lyapunov differential equation to approximate
state covariance along the trajectory
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
        dyn_func=lambda time, states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_m(
        ocp, nlp, n_noised_states=2, n_collocation_points=nlp.model.polynomial_degree + 1
    )
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=2)
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, stochastic_variables, nlp, with_noise=True
        ),
        allow_free_variables=True,
    )


def cost(controller: PenaltyController):
    q = controller.states["q"].cx_start
    return (q - 3) ** 2


def bound(t):
    return 2 + 0.1 * cas.cos(10 * t)


# def path_constraint(controller: PenaltyController, dt, is_robustified: bool = False):
def path_constraint(controller, dt, is_robustified: bool = False):
    t = controller.time.cx_start  # controller.node_index * dt
    q = controller.states["q"].cx_start
    sup = bound(t)
    if is_robustified:
        P = StochasticBioModel.reshape_to_matrix(
            controller.stochastic_variables["cov"].cx_start, controller.model.matrix_shape_cov
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
        dt=final_time / n_shooting,
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
            dynamic_function=lambda time, states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model.dynamics(
                states,
                controls,
                parameters,
                stochastic_variables,
                nlp,
                with_noise=with_noise,
            ),
            expand=True,
        )

        s_init = InitialGuessList()
        s_init.add(
            "m",
            initial_guess=[0] * bio_model.matrix_shape_m[0] * bio_model.matrix_shape_m[1],
            interpolation=InterpolationType.CONSTANT,
        )

        cov0 = np.diag([0.01**2, 0.1**2]).reshape((-1,), order="F")
        s_init.add(
            "cov",
            initial_guess=cov0,
            interpolation=InterpolationType.CONSTANT,
        )
        constraints.add(ConstraintFcn.TRACK_STOCHASTIC, key="cov", node=Node.START, target=cov0)

        return StochasticOptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            s_init=s_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            n_threads=1,
            assume_phase_dynamics=True,
            problem_type=problem_type,
        )

    else:
        dynamics.add(
            configure_optimal_control_problem,
            dynamic_function=lambda time, states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model.dynamics(
                states,
                controls,
                parameters,
                stochastic_variables,
                nlp,
                with_noise=with_noise,
            ),
            expand=True,
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
            assume_phase_dynamics=True,
        )


def main():
    """
    Prepare, solve and plot the solution
    """
    isStochastic = True
    isRobust = True
    if not isStochastic:
        isRobust = False

    # --- Prepare the ocp --- #
    d = 5  # polynomial_degree
    socp_type = SocpType.COLLOCATION(polynomial_degree=d, method="legendre")
    bio_model = RockitModel(socp_type=socp_type)
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
        polynomial_degree=d,
        motor_noise_magnitude=motor_noise_magnitude,
        is_stochastic=isStochastic,
        is_robustified=isRobust,
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

    if isStochastic:
        cov = sol_socp.stochastic_variables["cov"]

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
            np.squeeze([q[:, :: d + 2] - sigma, q[:, :: d + 2] + sigma]),
            "k",
        )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
