import importlib.util
from pathlib import Path

import biorbd
import numpy as np
from casadi import MX, horzcat

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    ShowResult,
    Data,
    ObjectiveList,
    Objective,
    Axe,
    PlotType,
    OdeSolver,
)

# Load align_segment_on_rt
EXAMPLES_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location("data_to_track", str(EXAMPLES_FOLDER) + "/getting_started/pendulum.py")
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def get_markers_pos(x, idx_coord, fun):
    marker_pos = []
    for i in range(x.shape[1]):
        marker_pos.append(fun(x[:nb_q, i]))
    marker_pos = horzcat(*marker_pos)
    return marker_pos[:, idx_coord]


def prepare_ocp(biorbd_model, final_time, number_shooting_points, markers_ref, tau_ref, ode_solver=OdeSolver.RK):
    # --- Options --- #
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        Objective.Lagrange.TRACK_MARKERS, axis_to_track=[Axe.Y, Axe.Z], weight=100, target=markers_ref
    )
    objective_functions.add(Objective.Lagrange.TRACK_TORQUE, target=tau_ref)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (n_q + n_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * n_tau, [tau_max] * n_tau])

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    biorbd_path = str(EXAMPLES_FOLDER) + "/getting_started/pendulum.bioMod"
    biorbd_model = biorbd.Model(biorbd_path)
    final_time = 3
    number_shooting_points = 20

    ocp_to_track = data_to_track.prepare_ocp(
        biorbd_model_path=biorbd_path, final_time=3, number_shooting_points=19, nb_threads=4
    )
    sol_to_track = ocp_to_track.solve()
    states, controls = Data.get_data(ocp_to_track, sol_to_track)
    q, q_dot, tau = states["q"], states["q_dot"], controls["tau"]
    nb_q = nb_qdot = nb_tau = biorbd_model.nbQ()
    nb_marker = biorbd_model.nbMarkers()
    x = np.concatenate((q, q_dot))
    u = tau

    symbolic_states = MX.sym("q", nb_q, 1)
    symbolic_controls = MX.sym("u", nb_tau, 1)
    markers_fun = biorbd.to_casadi_func("ForwardKin", biorbd_model.markers, symbolic_states)
    markers_ref = np.zeros((3, nb_marker, number_shooting_points + 1))
    for i in range(number_shooting_points):
        markers_ref[:, :, i] = markers_fun(x[:nb_q, i])
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
        label_markers.append(ocp.nlp[0].model.markerNames()[mark].to_string())

    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u, p: get_markers_pos(x, 0, markers_fun),
        plot_type=PlotType.PLOT,
        color="tab:red",
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u, p: markers_ref[1, :, :],
        plot_type=PlotType.STEP,
        color="black",
        legend=label_markers,
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u, p: get_markers_pos(x, 1, markers_fun),
        plot_type=PlotType.PLOT,
        color="tab:green",
    )
    ocp.add_plot(
        "Markers plot coordinates",
        update_function=lambda x, u, p: markers_ref[2, :, :],
        plot_type=PlotType.STEP,
        color="black",
        legend=label_markers,
    )
    # ocp.add_plot(
    #     "Markers plot coordinates",
    #     update_function=lambda x, u, p: get_markers_pos(x, 2, markers_fun),
    #     plot_type=PlotType.PLOT,
    #     color="tab:blue",
    # )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
