"""
This example aims to replicate the example provided in Gillis 2013: https://doi.org/10.1109/CDC.2013.6761121.
It consists in a mass-point trying to find a time optimal periodic trajectory around super-ellipse obstacles.
The controls are coordinates of a quide-point (the mass is attached to this guide point with a sping).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import casadi as cas
import numpy as np

from bioptim import (
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    OptimalControlProgram,
    NonLinearProgram,
    DynamicsList,
    InterpolationType,
    SocpType,
    Node,
    ConstraintList,
    InitialGuessList,
    ControlType,
    PenaltyController,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ConfigureProblem,
    ConstraintFcn,
    StochasticBioModel,
    OdeSolver,
    PhaseDynamics,
    BoundsList,
    SolutionMerge,
)

from bioptim.examples.stochastic_optimal_control.models.mass_point_model import MassPointModel
from bioptim.examples.stochastic_optimal_control.common import (
    test_matrix_semi_definite_positiveness,
    test_eigen_values,
    reshape_to_matrix,
)

from scipy.integrate import solve_ivp

def superellipse(a=1, b=1, n=2, x_0=0, y_0=0, resolution=100):
    x = np.linspace(-2 * a + x_0, 2 * a + x_0, resolution)
    y = np.linspace(-2 * b + y_0, 2 * b + y_0, resolution)

    X, Y = np.meshgrid(x, y)
    Z = ((X - x_0) / a) ** n + ((Y - y_0) / b) ** n - 1
    return X, Y, Z


def draw_cov_ellipse(cov, pos, ax, color="b"):
    """
    Draw an ellipse representing the covariance at a given point.
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(vals)
    ellip = plt.matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, color=color, alpha=0.1)

    ax.add_patch(ellip)
    return ellip


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
        ocp, nlp, n_noised_states=4, n_collocation_points=nlp.model.polynomial_degree + 1
    )
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=4)
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
    )


def path_constraint(controller: PenaltyController, super_elipse_index: int, is_robustified: bool = False):
    p_x = controller.states["q"].cx[0]
    p_y = controller.states["q"].cx[1]

    h = (
        (
            (p_x - controller.model.super_ellipse_center_x[super_elipse_index])
            / controller.model.super_ellipse_a[super_elipse_index]
        )
        ** controller.model.super_ellipse_n[super_elipse_index]
        + (
            (p_y - controller.model.super_ellipse_center_y[super_elipse_index])
            / controller.model.super_ellipse_b[super_elipse_index]
        )
        ** controller.model.super_ellipse_n[super_elipse_index]
        - 1
    )

    out = h

    if is_robustified:
        gamma = 1
        dh_dx = cas.jacobian(h, controller.states.cx)
        cov = StochasticBioModel.reshape_to_matrix(
            controller.algebraic_states["cov"].cx, controller.model.matrix_shape_cov
        )
        safe_guard = gamma * cas.sqrt(dh_dx @ cov @ dh_dx.T)
        out -= safe_guard

    return out


def initialize_circle(n_points):
    """
    Initialize the positions equally distributed over a circle of radius 3
    """
    q_init = np.zeros((2, n_points))
    for i in range(n_points):
        q_init[0, i] = 3 * np.sin(i * 2 * np.pi / (n_points - 1))
        q_init[1, i] = 3 * np.cos(i * 2 * np.pi / (n_points - 1))
    return q_init


def prepare_socp(
    final_time: float,
    n_shooting: int,
    motor_noise_magnitude: np.ndarray,
    polynomial_degree: int,
    q_init: np.ndarray,
    is_stochastic: bool = False,
    is_robustified: bool = False,
    socp_type=SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
    expand_dynamics: bool = True,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    use_sx: bool = False,
) -> StochasticOptimalControlProgram | OptimalControlProgram:
    problem_type = socp_type

    bio_model = MassPointModel(
        socp_type=problem_type, motor_noise_magnitude=motor_noise_magnitude, polynomial_degree=polynomial_degree
    )

    nb_q = bio_model.nb_q
    nb_qdot = bio_model.nb_qdot
    nb_u = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTROL,
        key="u",
        weight=1e-2 / (2 * n_shooting),
        node=Node.ALL_SHOOTING,
        quadratic=True,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        path_constraint,
        node=Node.ALL,
        super_elipse_index=0,
        min_bound=0,
        max_bound=cas.inf,
        is_robustified=is_robustified,
        quadratic=False,
    )
    constraints.add(
        path_constraint,
        node=Node.ALL,
        super_elipse_index=1,
        min_bound=0,
        max_bound=cas.inf,
        is_robustified=is_robustified,
        quadratic=False,
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=0, node=Node.START, target=0)
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    # Initial guesses
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.ALL_POINTS)
    x_init.add("qdot", initial_guess=[0] * nb_qdot, interpolation=InterpolationType.CONSTANT)

    control_init = InitialGuessList()
    control_init.add("u", initial_guess=[0] * nb_u, interpolation=InterpolationType.CONSTANT)

    # Initial bounds
    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=[-10, -10], max_bound=[10, 10], interpolation=InterpolationType.CONSTANT)
    x_bounds.add("qdot", min_bound=[-20, -20], max_bound=[20, 20], interpolation=InterpolationType.CONSTANT)

    control_bounds = BoundsList()
    control_bounds.add("u", min_bound=[-20, -20], max_bound=[20, 20], interpolation=InterpolationType.CONSTANT)

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

        phase_transitions.add(PhaseTransitionFcn.COVARIANCE_CYCLIC)

        a_init = InitialGuessList()
        a_init.add(
            "m",
            initial_guess=[0] * bio_model.matrix_shape_m[0] * bio_model.matrix_shape_m[1],
            interpolation=InterpolationType.CONSTANT,
        )

        cov0 = (np.eye(bio_model.matrix_shape_cov[0]) * 0.01).reshape((-1,), order="F")
        a_init.add(
            "cov",
            initial_guess=cov0,
            interpolation=InterpolationType.CONSTANT,
        )

        return StochasticOptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init=x_init,
            u_init=control_init,
            a_init=a_init,
            x_bounds=x_bounds,
            u_bounds=control_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            control_type=ControlType.CONSTANT,
            n_threads=6,
            problem_type=problem_type,
            phase_transitions=phase_transitions,
            use_sx=use_sx,
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
        ode_solver = OdeSolver.COLLOCATION(
            polynomial_degree=socp_type.polynomial_degree,
            method=socp_type.method,
            duplicate_starting_point=True,
        )

        return OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init=x_init,
            u_init=control_init,
            x_bounds=x_bounds,
            u_bounds=control_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            control_type=ControlType.CONSTANT,
            n_threads=6,
            phase_transitions=phase_transitions,
            ode_solver=ode_solver,
        )


def main():
    """
    Prepare, solve and plot the solution
    """

    use_sx = True
    is_stochastic = True
    is_robust = True
    if not is_stochastic:
        is_robust = False

    polynomial_degree = 5

    # --- Prepare the ocp --- #
    socp_type = SocpType.COLLOCATION(polynomial_degree=polynomial_degree, method="legendre")

    n_shooting = 40
    final_time = 4
    motor_noise_magnitude = np.array([50, 50])
    bio_model = MassPointModel(socp_type=socp_type, motor_noise_magnitude=motor_noise_magnitude)

    q_init = np.zeros((bio_model.nb_q, (polynomial_degree + 2) * n_shooting + 1))
    zq_init = initialize_circle((polynomial_degree + 1) * n_shooting + 1)
    for i in range(n_shooting + 1):
        j = i * (polynomial_degree + 1)
        k = i * (polynomial_degree + 2)
        q_init[:, k] = zq_init[:, j]
        q_init[:, k + 1 : k + 1 + (polynomial_degree + 1)] = zq_init[:, j : j + (polynomial_degree + 1)]

    socp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        polynomial_degree=polynomial_degree,
        motor_noise_magnitude=motor_noise_magnitude,
        q_init=q_init,
        is_stochastic=is_stochastic,
        is_robustified=is_robust,
        socp_type=socp_type,
        use_sx=use_sx,
    )

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")

    # Check if the file exists
    import os, pickle
    filename = "obstacle.pkl"
    if os.path.exists(filename):
        # Open the file and load the content
        with open(filename, "rb") as file:
            data_loaded = pickle.load(file)
        # Extract variables from the loaded data
        time = data_loaded["time"]
        states = data_loaded["states"]
        controls = data_loaded["controls"]
        algebraic_states = data_loaded["algebraic_states"]
        print("File loaded successfully.")

    else:
        sol_socp = socp.solve(solver)

        time = sol_socp.decision_time(to_merge=SolutionMerge.NODES)
        states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
        algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

        data_to_save = {
            "time":  time,
            "states": states,
            "controls": controls,
            "algebraic_states": algebraic_states,
        }
        with open(filename, "wb") as file:
            pickle.dump(data_to_save, file)





    q = states["q"]
    qdot = states["qdot"]
    u = controls["u"]
    Tf = time[-1]
    tgrid = np.linspace(0, Tf, n_shooting + 1).squeeze()


    fig, ax = plt.subplots(2, 2)
    for i in range(2):
        a = bio_model.super_ellipse_a[i]
        b = bio_model.super_ellipse_b[i]
        n = bio_model.super_ellipse_n[i]
        x_0 = bio_model.super_ellipse_center_x[i]
        y_0 = bio_model.super_ellipse_center_y[i]

        X, Y, Z = superellipse(a, b, n, x_0, y_0)

        ax[0, 0].contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5)

    ax[0, 0].plot(q_init[0], q_init[1], "-k", label="Initial guess")
    ax[0, 0].plot(q[0][0], q[1][0], "og")
    ax[0, 0].plot(q[0], q[1], "-g", label="q")

    ax[0, 1].plot(q[0], q[1], "b")
    ax[0, 1].plot(u[0], u[1], "r")
    for i in range(n_shooting):
        ax[0, 1].plot((u[0][i], q[0][i * (polynomial_degree + 2)]), (u[1][i], q[1][i * (polynomial_degree + 2)]), ":k")

    ax[1, 0].plot(tgrid, q[0, :: polynomial_degree + 2], "--", label="px")
    ax[1, 0].plot(tgrid, q[1, :: polynomial_degree + 2], "-", label="py")
    ax[1, 0].step(tgrid[:-1], u.T, "-.", label="u")
    ax[1, 0].set_xlabel("t")

    if is_stochastic:
        m = algebraic_states["m"]
        cov = algebraic_states["cov"]

        # estimate covariance using series of noisy trials
        iter = 200
        np.random.seed(42)
        noise = np.vstack([
            np.random.normal(loc=0, scale=motor_noise_magnitude[0], size=(1, n_shooting, iter)),
            np.random.normal(loc=0, scale=motor_noise_magnitude[1], size=(1, n_shooting, iter))
            ])

        nx = bio_model.nb_q + bio_model.nb_qdot
        cov_numeric = np.empty((nx, nx, iter))
        x_mean = np.empty((nx, iter))
        x_std = np.empty((nx, iter))

        for i in range(n_shooting):
            x_i = np.hstack([
                q[:, i * (polynomial_degree + 2)],
                qdot[:, i * (polynomial_degree + 2)]
            ])#.T
            t_span = tgrid[i:i+2]

            next_x = np.empty((4, iter))
            for it in range(iter):
                dynamics = lambda t, x: bio_model.dynamics_numerical(
                    states=x,
                    controls=u[:, i].T,
                    motor_noise=noise[:, i, it].T
                ).full().T
                sol_ode = solve_ivp(dynamics, t_span, x_i, method='RK45')
                next_x[:, it] = sol_ode.y[:, -1]

            x_mean[:, i] = np.mean(next_x, axis=1)
            x_std[:, i] = np.std(next_x, axis=1)
            cov_numeric[:, :, i] = np.cov(next_x)
            ax[0, 0].plot(next_x[0, :], next_x[1, :], ".r")
            ax[0, 0].plot([x_mean[0, i], x_mean[0, i]],  x_mean[1, i] + [-x_std[1, i], x_std[1, i]], "-k")
            ax[0, 0].plot(x_mean[0, i] + [-x_std[0, i], x_std[0, i]], [x_mean[1, i], x_mean[1, i]], "-k")

            draw_cov_ellipse(cov_numeric[:2, :2, i], x_mean[:, i], ax[0, 0], color="r")


        ax[0, 0].plot(x_mean[0, :], x_mean[1, :], "+r")



        for i in range(n_shooting + 1):
            cov_i = cov[:, i]
            if not test_matrix_semi_definite_positiveness(cov_i):
                print(f"Something went wrong at the {i}th node. (Semi-definiteness)")

            if not test_eigen_values(cov_i):
                print(f"Something went wrong at the {i}th node. (Eigen values)")

            cov_i = reshape_to_matrix(cov_i, (bio_model.matrix_shape_cov))
            draw_cov_ellipse(cov_i[:2, :2], q[:, i * (polynomial_degree + 2)], ax[0, 0], color="b")
    plt.show()


if __name__ == "__main__":
    main()
