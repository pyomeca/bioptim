"""
This example aims to replicate the example provided in Gillis 2013: https://doi.org/10.1109/CDC.2013.6761121.
It consists in a mass-point trying to find a time optimal periodic trajectory around super-ellipse obstacles.
The controls are coordinates of a quide-point (the mass is attached to this guide point with a sping).
"""

import pickle

import casadi as cas
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp

from bioptim import (
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    OptimalControlProgram,
    DynamicsOptions,
    InterpolationType,
    SocpType,
    Node,
    ConstraintList,
    InitialGuessList,
    ControlType,
    PenaltyController,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ConstraintFcn,
    StochasticBioModel,
    OdeSolver,
    PhaseDynamics,
    BoundsList,
    SolutionMerge,
    SolutionIntegrator,
    Shooting,
)
from bioptim.examples.toy_examples.stochastic_optimal_control.common import (
    test_matrix_semi_definite_positiveness,
    test_eigen_values,
    reshape_to_matrix,
)
from bioptim.examples.toy_examples.stochastic_optimal_control.models.mass_point_model import (
    MassPointDynamicsModel,
    StochasticMassPointDynamicsModel,
)


def plot_results(
    sol_socp,
    states,
    controls,
    time,
    algebraic_states,
    bio_model,
    motor_noise_magnitude,
    n_shooting,
    polynomial_degree,
    is_stochastic,
    q_init,
):
    """
    This function plots the reintegration of the optimal solution considering the motor noise.
    The plot compares the covariance obtained numerically by doing 100 orbits, the covariance obtained by the optimal control problem and the covariance obtained by the noisy integration.
    """
    q = states["q"]
    qdot = states["qdot"]
    u = controls["u"]
    Tf = time[-1]
    tgrid = np.linspace(0, Tf, n_shooting + 1).squeeze()

    fig, ax = plt.subplots(2, 2)
    fig_comparison, ax_comparison = plt.subplots(1, 1)
    for i in range(2):
        a = bio_model.super_ellipse_a[i]
        b = bio_model.super_ellipse_b[i]
        n = bio_model.super_ellipse_n[i]
        x_0 = bio_model.super_ellipse_center_x[i]
        y_0 = bio_model.super_ellipse_center_y[i]

        X, Y, Z = superellipse(a, b, n, x_0, y_0)

        ax[0, 0].contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5, label="Obstacles")
        ax_comparison.contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5, label="Obstacles")

    ax[0, 0].plot(q_init[0], q_init[1], "-k", label="Initial guess")
    ax[0, 0].plot(q[0][0], q[1][0], "og", label="Optimal initial node")
    ax[0, 0].plot(q[0], q[1], "-g", label="Optimal trajectory")

    ax[0, 1].plot(q[0], q[1], "b", label="Optimal trajectory")
    ax[0, 1].plot(u[0], u[1], "r", label="Optimal controls")
    for i in range(n_shooting):
        if i == 0:
            ax[0, 1].plot(
                (u[0][i], q[0][i * (polynomial_degree + 2)]),
                (u[1][i], q[1][i * (polynomial_degree + 2)]),
                ":k",
                label="Spring orientation",
            )
        else:
            ax[0, 1].plot(
                (u[0][i], q[0][i * (polynomial_degree + 2)]), (u[1][i], q[1][i * (polynomial_degree + 2)]), ":k"
            )
    ax[0, 1].legend()

    ax[1, 0].step(tgrid, u.T, "-.", label=["Optimal controls X", "Optimal controls Y"])
    ax[1, 0].fill_between(
        tgrid,
        u.T[:, 0] - motor_noise_magnitude[0],
        u.T[:, 0] + motor_noise_magnitude[0],
        step="pre",
        alpha=0.3,
        color="#1f77b4",
    )
    ax[1, 0].fill_between(
        tgrid,
        u.T[:, 1] - motor_noise_magnitude[1],
        u.T[:, 1] + motor_noise_magnitude[1],
        step="pre",
        alpha=0.3,
        color="#ff7f0e",
    )

    ax[1, 0].plot(tgrid, q[0, :: polynomial_degree + 2], "--", label="Optimal trajectory X")
    ax[1, 0].plot(tgrid, q[1, :: polynomial_degree + 2], "-", label="Optimal trajectory Y")

    ax[1, 0].set_xlabel("Time [s]")
    ax[1, 0].legend()

    if is_stochastic:
        cov = controls["cov"]

        # estimate covariance using series of noisy trials
        iter = 200
        np.random.seed(42)
        noise = np.vstack(
            [
                np.random.normal(loc=0, scale=motor_noise_magnitude[0], size=(1, n_shooting, iter)),
                np.random.normal(loc=0, scale=motor_noise_magnitude[1], size=(1, n_shooting, iter)),
            ]
        )

        nx = bio_model.nb_q + bio_model.nb_qdot
        cov_numeric = np.zeros((nx, nx, n_shooting))
        x_mean = np.zeros((nx, n_shooting + 1))
        x_std = np.zeros((nx, n_shooting + 1))
        dt = Tf / (n_shooting)

        x_j = np.zeros((nx,))
        for i in range(n_shooting):
            x_i = np.hstack([q[:, i * (polynomial_degree + 2)], qdot[:, i * (polynomial_degree + 2)]])
            new_u = np.hstack([u[:, i:], u[:, :i]])
            next_x = np.zeros((nx, iter))
            for it in range(iter):

                x_j[:] = x_i[:]
                for j in range(n_shooting):
                    dynamics = (
                        lambda t, x: bio_model.dynamics_numerical(
                            states=x, controls=new_u[:, j].T, motor_noise=noise[:, j, it].T
                        )
                        .full()
                        .T
                    )
                    sol_ode = solve_ivp(dynamics, t_span=[0, dt], y0=x_j, method="RK45")
                    x_j[:] = sol_ode.y[:, -1]

                next_x[:, it] = x_j[:]

            x_mean[:, i] = np.mean(next_x, axis=1)
            x_std[:, i] = np.std(next_x, axis=1)

            cov_numeric[:, :, i] = np.cov(next_x)
            if i == 0:
                ax[0, 0].plot(next_x[0, :], next_x[1, :], ".r", label="Noisy integration")
            else:
                ax[0, 0].plot(next_x[0, :], next_x[1, :], ".r")
            # We can draw the X and Y covariance just for personnal reference, but the eigen vectors of the covariance matrix do not have to be aligned with the horizontal and vertical axis
            # ax[0, 0].plot([x_mean[0, i], x_mean[0, i]], x_mean[1, i] + [-x_std[1, i], x_std[1, i]], "-k", label="Numerical covariance")
            # ax[0, 0].plot(x_mean[0, i] + [-x_std[0, i], x_std[0, i]], [x_mean[1, i], x_mean[1, i]], "-k")
            if i == 0:
                draw_cov_ellipse(
                    cov_numeric[:2, :2, i], x_mean[:, i], ax[0, 0], color="r", label="Numerical covariance"
                )
                draw_cov_ellipse(
                    cov_numeric[:2, :2, i], x_mean[:, i], ax_comparison, color="r", label="Numerical covariance"
                )
            else:
                draw_cov_ellipse(cov_numeric[:2, :2, i], x_mean[:, i], ax[0, 0], color="r")
                draw_cov_ellipse(cov_numeric[:2, :2, i], x_mean[:, i], ax_comparison, color="r")

        ax[1, 0].fill_between(
            tgrid,
            q[0, :: polynomial_degree + 2] - x_std[0, :],
            q[0, :: polynomial_degree + 2] + x_std[0, :],
            alpha=0.3,
            color="#2ca02c",
        )

        ax[1, 0].fill_between(
            tgrid,
            q[1, :: polynomial_degree + 2] - x_std[1, :],
            q[1, :: polynomial_degree + 2] + x_std[1, :],
            alpha=0.3,
            color="#d62728",
        )

        ax[0, 0].plot(x_mean[0, :], x_mean[1, :], "+b", label="Numerical mean")

        for i in range(n_shooting + 1):
            cov_i = cov[:, i]
            if not test_matrix_semi_definite_positiveness(cov_i):
                print(f"Something went wrong at the {i}th node. (Semi-definiteness)")

            if not test_eigen_values(cov_i):
                print(f"Something went wrong at the {i}th node. (Eigen values)")

            cov_i = reshape_to_matrix(cov_i, (bio_model.matrix_shape_cov))
            if i == 0:
                draw_cov_ellipse(
                    cov_i[:2, :2], q[:, i * (polynomial_degree + 2)], ax[0, 0], color="y", label="Optimal covariance"
                )
                draw_cov_ellipse(
                    cov_i[:2, :2],
                    q[:, i * (polynomial_degree + 2)],
                    ax_comparison,
                    color="y",
                    label="Optimal covariance",
                )
            else:
                draw_cov_ellipse(cov_i[:2, :2], q[:, i * (polynomial_degree + 2)], ax[0, 0], color="y")
                draw_cov_ellipse(cov_i[:2, :2], q[:, i * (polynomial_degree + 2)], ax_comparison, color="y")
    ax[0, 0].legend()

    # Integrate the nominal dynamics (as if it was deterministic)
    integrated_sol = sol_socp.integrate(
        shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.SCIPY_RK45, to_merge=SolutionMerge.NODES
    )
    # Integrate the stochastic dynamics (considering the feedback and the motor and sensory noises)
    noisy_integrated_sol = sol_socp.noisy_integrate(
        integrator=SolutionIntegrator.SCIPY_RK45, to_merge=SolutionMerge.NODES
    )

    # compare with noisy integration
    cov_integrated = np.zeros((2, 2, n_shooting + 1))
    mean_integrated = np.zeros((2, n_shooting + 1))
    i_node = 0
    for i in range(noisy_integrated_sol["q"][0].shape[0]):
        if i == 0:
            ax_comparison.plot(
                noisy_integrated_sol["q"][0][i, :],
                noisy_integrated_sol["q"][1][i, :],
                ".",
                color=cm.viridis(i / noisy_integrated_sol["q"][0].shape[0]),
                alpha=0.1,
                label="Noisy integration",
            )
        else:
            ax_comparison.plot(
                noisy_integrated_sol["q"][0][i, :],
                noisy_integrated_sol["q"][1][i, :],
                ".",
                color=cm.viridis(i / noisy_integrated_sol["q"][0].shape[0]),
                alpha=0.1,
            )
        if i % 7 == 0:
            cov_integrated[:, :, i_node] = np.cov(
                np.vstack((noisy_integrated_sol["q"][0][i, :], noisy_integrated_sol["q"][1][i, :]))
            )
            mean_integrated[:, i_node] = np.mean(
                np.vstack((noisy_integrated_sol["q"][0][i, :], noisy_integrated_sol["q"][1][i, :])), axis=1
            )

            if i == 0:
                draw_cov_ellipse(
                    cov_integrated[:2, :2, i_node],
                    mean_integrated[:, i_node],
                    ax_comparison,
                    color="b",
                    label="Noisy integration covariance",
                )
            else:
                draw_cov_ellipse(
                    cov_integrated[:2, :2, i_node],
                    mean_integrated[:, i_node],
                    ax_comparison,
                    color="b",
                )
            i_node += 1
    ax_comparison.legend()
    fig_comparison.tight_layout()
    fig_comparison.savefig("comparison.png")
    plt.show()


def superellipse(a=1, b=1, n=2, x_0=0, y_0=0, resolution=100):
    x = np.linspace(-2 * a + x_0, 2 * a + x_0, resolution)
    y = np.linspace(-2 * b + y_0, 2 * b + y_0, resolution)

    X, Y = np.meshgrid(x, y)
    Z = ((X - x_0) / a) ** n + ((Y - y_0) / b) ** n - 1
    return X, Y, Z


def draw_cov_ellipse(cov, pos, ax, **kwargs):
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
    ellip = plt.matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=0.5, **kwargs)

    ax.add_patch(ellip)
    return ellip


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
        cov = StochasticBioModel.reshape_to_matrix(controller.controls["cov"].cx, controller.model.matrix_shape_cov)
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

    if is_stochastic:
        bio_model = StochasticMassPointDynamicsModel(
            problem_type=socp_type, motor_noise_magnitude=motor_noise_magnitude, polynomial_degree=polynomial_degree
        )
    else:
        bio_model = MassPointDynamicsModel(
            problem_type=socp_type, motor_noise_magnitude=motor_noise_magnitude, polynomial_degree=polynomial_degree
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
    if is_stochastic:
        u_min = np.ones((nb_u, n_shooting + 1)) * -20
        u_min[:, -1] = 0
        u_max = np.ones((nb_u, n_shooting + 1)) * 20
        u_max[:, -1] = 0
        control_bounds.add("u", min_bound=u_min, max_bound=u_max, interpolation=InterpolationType.EACH_FRAME)
    else:
        control_bounds.add("u", min_bound=[-20, -20], max_bound=[20, 20], interpolation=InterpolationType.CONSTANT)

    a_bounds = BoundsList()
    a_bounds.add(
        "m",
        min_bound=np.ones((bio_model.matrix_shape_cov[0] * bio_model.matrix_shape_cov[0],)) * -cas.inf,
        max_bound=np.ones((bio_model.matrix_shape_cov[0] * bio_model.matrix_shape_cov[0],)) * cas.inf,
        interpolation=InterpolationType.CONSTANT,
    )

    if is_stochastic:

        # Dynamics
        dynamics = DynamicsOptions(
            phase_dynamics=phase_dynamics,
            expand_dynamics=expand_dynamics,
        )

        phase_transitions.add(PhaseTransitionFcn.COVARIANCE_CYCLIC)

        a_init = InitialGuessList()
        a_init.add(
            "m",
            initial_guess=[0] * bio_model.matrix_shape_cov[0] * bio_model.matrix_shape_cov[1],
            interpolation=InterpolationType.CONSTANT,
        )

        cov0 = (np.eye(bio_model.matrix_shape_cov[0]) * 0.01).reshape((-1,), order="F")
        control_init.add(
            "cov",
            initial_guess=cov0,
            interpolation=InterpolationType.CONSTANT,
        )

        return StochasticOptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_init=x_init,
            u_init=control_init,
            a_init=a_init,
            x_bounds=x_bounds,
            u_bounds=control_bounds,
            a_bounds=a_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            control_type=ControlType.CONSTANT_WITH_LAST_NODE,
            n_threads=6,
            problem_type=socp_type,
            phase_transitions=phase_transitions,
            use_sx=use_sx,
        )

    else:
        ode_solver = OdeSolver.COLLOCATION(
            polynomial_degree=socp_type.polynomial_degree,
            method=socp_type.method,
            duplicate_starting_point=True,
        )

        # Dynamics
        dynamics = DynamicsOptions(
            phase_dynamics=phase_dynamics,
            expand_dynamics=expand_dynamics,
            ode_solver=ode_solver,
        )

        return OptimalControlProgram(
            bio_model,
            n_shooting,
            final_time,
            dynamics=dynamics,
            x_init=x_init,
            u_init=control_init,
            x_bounds=x_bounds,
            u_bounds=control_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            control_type=ControlType.CONSTANT,
            n_threads=6,
            phase_transitions=phase_transitions,
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
    motor_noise_magnitude = np.array([1, 1]) * 1
    bio_model = MassPointDynamicsModel(problem_type=socp_type, motor_noise_magnitude=motor_noise_magnitude)

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
    # solver.set_linear_solver("ma57")
    sol_socp = socp.solve(solver)

    time = sol_socp.decision_time(to_merge=SolutionMerge.NODES)
    states = sol_socp.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol_socp.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol_socp.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    data_to_save = {
        "time": time,
        "states": states,
        "controls": controls,
        "algebraic_states": algebraic_states,
    }
    with open("obstacle.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

    plot_results(
        sol_socp,
        states,
        controls,
        time,
        algebraic_states,
        bio_model,
        motor_noise_magnitude,
        n_shooting,
        polynomial_degree,
        is_stochastic,
        q_init,
    )


if __name__ == "__main__":
    main()
