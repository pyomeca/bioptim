"""
This is a basic example on how to use inverse optimal control to recover the weightings of the objective functions.
The example is not well tuned, but it can be used as an example for your more meaningful problems.

Please note that this example is dependent on the external library Pygmo which can be installed through
conda install -c conda-forge pygmo
"""

import numpy as np
import matplotlib.pyplot as plt

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    Node,
    CostType,
    BiorbdModel,
    BiMappingList,
)


def prepare_ocp(weights, coefficients, biorbd_model_path="models/double_pendulum.bioMod"):
    # Parameters of the problem
    biorbd_model = BiorbdModel(biorbd_model_path)
    phase_time = 1.5
    n_shooting = 30
    tau_min, tau_max, tau_init = -100, 100, 0

    # Mapping to remove the actuation
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1])

    # Add objective functions
    objective_functions = ObjectiveList()
    if coefficients[0] * weights[0] != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=coefficients[0] * weights[0])
    if coefficients[1] * weights[1] != 0:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=coefficients[1] * weights[1]
        )
    if coefficients[2] * weights[2] != 0:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_MARKERS,
            node=Node.ALL_SHOOTING,
            derivative=True,
            weight=coefficients[2] * weights[2],
        )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    n_q = biorbd_model.nb_q
    n_qdot = n_q

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][0, 0] = -np.pi
    x_bounds[0][1, 0] = 0
    x_bounds[0][3, 0] = 5 * np.pi
    x_bounds[0][0, 2] = np.pi
    x_bounds[0][1, 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * n_q + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min], [tau_max])

    u_init = InitialGuessList()
    u_init.add([tau_init])

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        variable_mappings=tau_mappings,
        n_threads=4,
        assume_phase_dynamics=True,
    )


class prepare_iocp:
    """
    This class must be defined by the user to match the data to track.
    It must containt the following methods:
        - fitness: The function returning the fitness of the solution
        - get_nobj: The function returning the number of objectives
        - get_bounds: The function returning the bounds on the weightings
    """

    def __init__(self, coefficients, solver, q_to_track, qdot_to_track, tau_to_track):
        self.coefficients = coefficients
        self.solver = solver
        self.q_to_track = q_to_track
        self.qdot_to_track = qdot_to_track
        self.tau_to_track = tau_to_track

    def fitness(self, weights):
        """
        This function returns how well did the weightings allow to fit the data to track.
        The OCP is solved in this function.
        """
        global i_inverse
        i_inverse += 1
        ocp = prepare_ocp(weights, self.coefficients)
        sol = ocp.solve(self.solver)
        print(
            f"+++++++++++++++++++++++++++ Optimized the {i_inverse}th ocp in the inverse algo +++++++++++++++++++++++++++"
        )
        if sol.status == 0:
            q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
            return [
                np.sum((self.q_to_track - q) ** 2)
                + np.sum((self.qdot_to_track - qdot) ** 2)
                + np.sum((self.tau_to_track[:, :-1] - tau[:, :-1]) ** 2)
            ]
        else:
            return [1000000]

    def get_nobj(self):
        return 1

    def get_bounds(self):
        return ([0, 0, 0], [1, 1, 1])


def main():
    import pygmo as pg

    # Generate data using OCP
    weights_to_track = [0.4, 0.3, 0.3]
    ocp_to_track = prepare_ocp(weights=weights_to_track, coefficients=[1, 1, 1])
    ocp_to_track.add_plot_penalty(CostType.ALL)
    solver = Solver.IPOPT()
    solver.set_linear_solver("ma57")
    sol_to_track = ocp_to_track.solve(solver)
    q_to_track, qdot_to_track, tau_to_track = (
        sol_to_track.states["q"],
        sol_to_track.states["qdot"],
        sol_to_track.controls["tau"],
    )
    print("+++++++++++++++++++++++++++ weights_to_track generated +++++++++++++++++++++++++++")

    # Find coefficients of the objective using Pareto
    coefficients = []
    for i in range(len(weights_to_track)):
        weights_pareto = [0, 0, 0]
        weights_pareto[i] = 1
        ocp_pareto = prepare_ocp(weights=weights_pareto, coefficients=[1, 1, 1])
        sol_pareto = ocp_pareto.solve(solver)
        # sol_pareto.animate()
        coefficients.append(sol_pareto.cost)
    print("+++++++++++++++++++++++++++ coefficients generated +++++++++++++++++++++++++++")

    # Retrieving weights using IOCP
    global i_inverse
    i_inverse = 0
    iocp = pg.problem(prepare_iocp(coefficients, solver, q_to_track, qdot_to_track, tau_to_track))
    algo = pg.algorithm(pg.simulated_annealing())
    pop = pg.population(iocp, size=100)

    epsilon = 1e-8
    diff = 10000
    while i_inverse < 100000 and diff > epsilon:
        olf_pop_f = np.min(pop.get_f())
        pop = algo.evolve(pop)
        diff = olf_pop_f - np.min(pop.get_f())
        pop_weights = pop.get_x()[np.argmin(pop.get_f())]

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        "The optimizaed weight are : ",
        pop_weights[0] * coefficients[0],
        "/",
        pop_weights[1] * coefficients[1],
        "/",
        pop_weights[2] * coefficients[2],
    )
    print(
        "The tracked weight are : ",
        weights_to_track[0] * coefficients[0],
        "/",
        weights_to_track[1] * coefficients[1],
        "/",
        weights_to_track[2] * coefficients[2],
    )
    print(
        "The weight difference is : ",
        (pop_weights[0] - weights_to_track[0]) / weights_to_track[0] * 100,
        "% / ",
        (pop_weights[1] - weights_to_track[1]) / weights_to_track[1] * 100,
        "% / ",
        (pop_weights[2] - weights_to_track[2]) / weights_to_track[2] * 100,
        "%",
    )

    # Compare the kinematics
    import biorbd

    ocp_final = prepare_ocp(weights=pop_weights, coefficients=coefficients)
    sol_final = ocp_final.solve(solver)
    q_final, qdot_final, tau_final = sol_final.states["q"], sol_final.states["qdot"], sol_final.controls["tau"]

    m = biorbd.Model("models/double_pendulum.bioMod")
    markers_to_track = np.zeros((2, np.shape(q_to_track)[1], 3))
    markers_final = np.zeros((2, np.shape(q_to_track)[1], 3))
    for i in range(np.shape(q_to_track)[1]):
        markers_to_track[0, i, :] = m.markers(q_to_track[:, i])[1].to_array()
        markers_to_track[1, i, :] = m.markers(q_to_track[:, i])[3].to_array()
        markers_final[0, i, :] = m.markers(q_final[:, i])[1].to_array()
        markers_final[1, i, :] = m.markers(q_final[:, i])[3].to_array()

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(markers_to_track[0, :, 1], markers_to_track[0, :, 2], "-r", label="Tracked reference")
    axs[1].plot(markers_to_track[1, :, 1], markers_to_track[1, :, 2], "-r", label="Tracked reference")
    axs[0].plot(markers_final[0, :, 1], markers_final[0, :, 2], "--b", label="Solution with optimal weightings")
    axs[1].plot(markers_final[1, :, 1], markers_final[1, :, 2], "--b", label="Solution with optimal weightings")
    axs[0].legend()
    axs[0].set_title(
        "Marker trajectory of the reference problem and the final solution generated with the optimal solutions."
    )
    axs[0].set_xlabel("y [m]")
    axs[0].set_ylabel("Z [m]")
    axs[1].set_xlabel("y [m]")
    axs[1].set_ylabel("Z [m]")
    plt.show()

    sol_to_track.animate()
    sol_final.animate()


if __name__ == "__main__":
    main()
