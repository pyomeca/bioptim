"""
This is a basic example on how to use inverse optimal control to recover the weightings from an optimal reaching task.

Please note that this example is dependent on the external library Pygmo which can be installed through
conda install -c conda-forge pygmo
"""

import pygmo as pg
import numpy as np
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
    ConstraintList,
    ConstraintFcn,
    Node,
    CostType,
    BiorbdModel,
)

def prepare_ocp(weights, coefficients):

    # Parameters of the problem
    biorbd_model_path = "models/double_pendulum.bioMod"
    biorbd_model = (BiorbdModel(biorbd_model_path))
    phase_time = 1.5
    n_shooting = 30
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    if coefficients[0] * weights[0] != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=coefficients[0]*weights[0])
    if coefficients[1] * weights[1] != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=coefficients[1]*weights[1])
    if coefficients[2] * weights[2] != 0:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, node=Node.ALL_SHOOTING, derivative=True, weight=coefficients[2]*weights[2])
    if coefficients[3] * weights[3] != 0:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.5, max_bound=5, weight=coefficients[3]*weights[3])

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    n_q = biorbd_model.nb_q
    n_qdot = n_q

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][0, 0] = 0
    x_bounds[0][1, 0] = 0
    x_bounds[0][0, 2] = np.pi
    x_bounds[0][1, 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * n_q + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nb_tau,
        [tau_max] * biorbd_model.nb_tau,
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nb_tau)

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
        n_threads=4,
    )


class prepare_iocp:
    def __init__(self, coefficients, solver, q_to_track, qdot_to_track, tau_to_track):
        self.coefficients = coefficients
        self.solver = solver
        self.q_to_track = q_to_track
        self.qdot_to_track = qdot_to_track
        self.tau_to_track = tau_to_track

    def fitness(self, weights):
        global i_inverse
        i_inverse += 1
        ocp = prepare_ocp(weights, self.coefficients)
        # ocp.add_plot_penalty(CostType.ALL)
        sol = ocp.solve(self.solver)
        print(f"+++++++++++++++++++++++++++ Optimized the {i_inverse}th ocp in the inverse algo +++++++++++++++++++++++++++")
        if sol.status == 0:
            q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
            return [np.sum((self.q_to_track - q) ** 2) + np.sum((self.qdot_to_track - qdot) ** 2) + np.sum((self.tau_to_track[:, :-1] - tau[:, :-1]) ** 2)]
        else:
            return [1000000]

    def get_nobj(self):
        return 1

    def get_bounds(self):
        return ([0, 0, 0, 0], [1, 1, 1, 1])


def main():

    # Generate data using OCP
    weights_to_track = [0.4, 0.3, 0.2, 0.1]
    ocp_to_track = prepare_ocp(weights=weights_to_track, coefficients=[1, 1, 1, 1])
    ocp_to_track.add_plot_penalty(CostType.ALL)
    solver = Solver.IPOPT()
    solver.set_linear_solver("ma57")
    # solver.set_print_level(0)
    sol_to_track = ocp_to_track.solve(solver)
    q_to_track, qdot_to_track, tau_to_track = sol_to_track.states["q"], sol_to_track.states["qdot"], sol_to_track.controls["tau"]
    print("+++++++++++++++++++++++++++ weights_to_track generated +++++++++++++++++++++++++++")
    # sol_to_track.animate()

    # Find coefficients of the objective using Pareto
    coefficients = []
    for i in range(len(weights_to_track)):
        weights_pareto = [0, 0, 0, 0]
        weights_pareto[i] = 1
        ocp_pareto = prepare_ocp(weights=weights_pareto, coefficients=[1, 1, 1, 1])
        sol_pareto = ocp_pareto.solve(solver)
        # sol_pareto.animate()
        coefficients.append(sol_pareto.cost)
    print("+++++++++++++++++++++++++++ coefficients generated +++++++++++++++++++++++++++")

    # Retrieving weights using IOCP
    global i_inverse
    i_inverse = 0
    coefficients = [0.1, 0.2, 0.3, 0.4]
    iocp = pg.problem(prepare_iocp(coefficients, solver, q_to_track, qdot_to_track, tau_to_track))
    algo = pg.algorithm(pg.simulated_annealing())
    pop = pg.population(iocp, size=50)

    epsilon = 1e-8
    diff = 100000
    while i_inverse < 10000 and diff > epsilon:
        olf_pop_f = np.min(pop.get_f())
        pop = algo.evolve(pop)
        diff = olf_pop_f - np.min(pop.get_f())
        pop_weights = pop.get_x()[np.argmin(pop.get_f())]

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("The weight are : ", pop_weights)
    print("The coefficients are: ", coefficients)
    print("The weight difference is : ", pop_weights[0] * coefficients[0] - weights_to_track[0] * coefficients[0], ' / ',
                                         pop_weights[1] * coefficients[1] - weights_to_track[1] * coefficients[1], ' / ',
                                         pop_weights[2] * coefficients[2] - weights_to_track[2] * coefficients[2], ' / ',
                                         pop_weights[3] * coefficients[3] - weights_to_track[3] * coefficients[3])

if __name__ == "__main__":
    main()
