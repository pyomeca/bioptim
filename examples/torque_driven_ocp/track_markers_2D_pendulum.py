import importlib.util
from pathlib import Path

import biorbd
import numpy as np
from casadi import Function, MX, vertcat

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
    Objective,
    Axe,
    PlotType,
)

# Load align_segment_on_rt
EXAMPLES_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location("data_to_track", str(EXAMPLES_FOLDER) + "/getting_started/pendulum.py")
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def get_markers_pos(x, idx_coord, fun):
    marker_pos = []
    for i in range(x.shape[1]):
        marker_pos.append(fun(x[:, i]))
    marker_pos = vertcat(*marker_pos)
    return marker_pos[idx_coord::3, :].T


def prepare_ocp(biorbd_model, final_time, number_shooting_points, markers_ref, tau_ref):
    # --- Options --- #
    torque_min, torque_max, torque_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = (
        {
            "type": Objective.Lagrange.TRACK_MARKERS,
            "axis_tot_track": [Axe.Y, Axe.Z],
            "weight": 100,
            "data_to_track": markers_ref,
        },
        {"type": Objective.Lagrange.TRACK_TORQUE, "weight": 1, "data_to_track": tau_ref.T},
    )

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, 0] = 0
    X_bounds.max[:, 0] = 0

    # Initial guess
    X_init = InitialConditions([0] * (n_q + n_qdot))

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)

    U_init = InitialConditions([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
    )


if __name__ == "__main__":
    biorbd_path = str(EXAMPLES_FOLDER) + "/getting_started/pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_path)
    final_time = 3
    number_shooting_points = 20

    ocp_to_track = data_to_track.prepare_ocp(
        biorbd_model_path=biorbd_path, final_time=3, number_shooting_points=20, nb_threads=4
    )
    sol_to_track = ocp_to_track.solve()
    states, controls = Data.get_data(ocp_to_track, sol_to_track)
    q, q_dot, tau = states["q"], states["q_dot"], controls["tau"]
    nb_q = nb_qdot = nb_tau = biorbd_model.nbQ()
    nb_marker = biorbd_model.nbMarkers()
    x = np.concatenate((q, q_dot))
    u = tau

    symbolic_states = MX.sym("x", nb_q + nb_qdot, 1)
    symbolic_controls = MX.sym("u", nb_tau, 1)
    markers_fun = Function(
        "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states[:nb_q])], ["q"], ["marker"],
    ).expand()
    markers_ref = np.zeros((3, nb_marker, number_shooting_points + 1))
    for i in range(number_shooting_points + 1):
        markers_ref[:, :, i] = np.array(markers_fun(x[:, i]))
    tau_ref = tau

    ocp = prepare_ocp(
        biorbd_model,
        final_time=final_time,
        number_shooting_points=number_shooting_points,
        markers_ref=markers_ref,
        tau_ref=tau_ref,
    )

    # --- plot markers position --- #
    label_markers = []
    title_markers = ["x", "y", "z"]
    for mark in range(biorbd_model.nbMarkers()):
        label_markers.append(ocp.nlp[0]["model"].markerNames()[mark].to_string())

    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u: get_markers_pos(x, 0, markers_fun),
        plot_type=PlotType.PLOT,
        color="tab:red",
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u: markers_ref[1, :, :],
        plot_type=PlotType.STEP,
        color="black",
        legend=label_markers,
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u: get_markers_pos(x, 1, markers_fun),
        plot_type=PlotType.PLOT,
        color="tab:green",
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u: markers_ref[2, :, :],
        plot_type=PlotType.STEP,
        color="black",
        legend=label_markers,
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u: get_markers_pos(x, 2, markers_fun),
        plot_type=PlotType.PLOT,
        color="tab:blue",
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
