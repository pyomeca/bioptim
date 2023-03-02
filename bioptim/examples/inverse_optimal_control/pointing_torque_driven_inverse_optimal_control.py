"""
This is a basic example on how to use inverse optimal control to recover the weightings from an optimal reaching task.

Please note that this example is dependent on the external library Platypus which can be installed through
conda install -c conda-forge platypus-opt
"""

# from platypus import NSGAII, Problem, Real
import pygmo as pg
from IPython import embed

import numpy as np
import biorbd_casadi as biorbd
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

# # Load track_segment_on_rt
# spec = importlib.util.spec_from_file_location(
#     "data_to_track", str(Path(__file__).parent) + "/contact_forces_inequality_constraint_muscle.py"
# )
# data_to_track = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(data_to_track)


def prepare_ocp(weights, coefficients):

    # Parameters of the problem
    biorbd_model_path = "models/multiple_pendulum.bioMod"
    biorbd_model = (BiorbdModel(biorbd_model_path))
    phase_time = 1.5
    n_shooting = 30
    tau_min, tau_max, tau_init = -25, 25, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    # Objective functions from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002183
    # hand jerk (minimize_marker_position_acceleration derivative=True) marker_acc biorbd (sinon velocity)
    # angle jerk (minimize_states_acceleration, derivative=True) dynamique inverse
    # angle acceleration (minimize_states_velocity, derivative=True)
    # Torque change (minimize_torques, derivative=True)
    if coefficients[0] * weights[0] != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=coefficients[0]*weights[0])
    if coefficients[1] * weights[0] != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=coefficients[1]*weights[0])
    # Effort/Snap (minimize_jerks, derivative=True)
    # Geodesic/hand trajectory (minimize_marker, derivative=True, mayer)
    # if coefficients[2] * weights[1] != 0:
    #     objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, node=Node.ALL_SHOOTING, derivative=True, weight=coefficients[2]*weights[1])
    # Energy (norm(qdot*tau))
    if coefficients[2] * weights[0] != 0:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.5, max_bound=5, weight=coefficients[2]*weights[2])

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, marker_index="marker_6", target=np.array([-0.0005*5, 0.0688*5, -0.9542*5]))
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.END, marker_index="marker_6", target=np.array([-0.0005*5, 0.0688*5, 0]))

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    n_q = biorbd_model.nb_q
    n_qdot = n_q

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_model.bounds_from_ranges(["q", "qdot"]))

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
        constraints=constraints,
        ode_solver=OdeSolver.RK4(),
        n_threads=4,
    )


# def prepare_iocp(weights, coefficients, solver, q_to_track, qdot_to_track, tau_to_track):
#     global i_inverse
#     i_inverse += 1
#     ocp = prepare_ocp(weights, coefficients)
#     # ocp.add_plot_penalty(CostType.ALL)
#     sol = ocp.solve(solver)
#     q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
#     print(f"+++++++++++++++++++++++++++ Optimized the {i_inverse}th ocp in the inverse algo +++++++++++++++++++++++++++")
#     return [np.sum((q_to_track - q) ** 2), np.sum((qdot_to_track - qdot) ** 2), np.sum((tau_to_track - tau) ** 2)]

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
        q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
        print(f"+++++++++++++++++++++++++++ Optimized the {i_inverse}th ocp in the inverse algo +++++++++++++++++++++++++++")
        # [np.sum((self.q_to_track - q) ** 2), np.sum((self.qdot_to_track - qdot) ** 2), np.sum((self.tau_to_track - tau) ** 2)]
        return [np.sum((self.q_to_track - q) ** 2) + np.sum((self.qdot_to_track - qdot) ** 2) + np.sum((self.tau_to_track[:, :-1] - tau[:, :-1]) ** 2)]

    def get_nobj(self):
        return 1

    def get_bounds(self):
        return ([0, 0, 0], [1, 1, 1])


def main():

    # # Generate data using OCP
    weights_to_track = [0.6, 0.3, 0.1]
    ocp_to_track = prepare_ocp(weights=weights_to_track, coefficients=[1, 1, 1])
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
        weights_pareto = [0, 0, 0]
        weights_pareto[i] = 1
        ocp_pareto = prepare_ocp(weights=weights_pareto, coefficients=[1, 1, 1])
        sol_pareto = ocp_pareto.solve(solver)
        sol_pareto.print()
        # sol_pareto.animate()
        coefficients.append(sol_pareto.cost)
    print("+++++++++++++++++++++++++++ coefficients generated +++++++++++++++++++++++++++")

    # Retrieving weights using IOCP
    global i_inverse
    i_inverse = 0
    # iocp = Problem(3, 3) # number of decision variables = 3, number of objectives = 3
    # iocp.types[:] = [Real(0, 1), Real(0, 1), Real(0, 1)]
    # iocp.function = lambda weights: prepare_iocp(weights, coefficients, solver, q_to_track, qdot_to_track, tau_to_track)
    # algorithm = NSGAII(iocp)
    # algorithm.run(1000)
    # result = algorithm.result
    # weights_optimized = result[-1].variables
    coefficients = [0.1, 0.4, 0.5]
    iocp = pg.problem(prepare_iocp(coefficients, solver, q_to_track, qdot_to_track, tau_to_track))
    algo = pg.algorithm(pg.simulated_annealing())
    pop = pg.population(iocp, size=50)

    epsilon = 1e-6
    diff = 1000
    while i_inverse < 1 and diff > epsilon: # 1000
        olf_pop_f = np.min(pop.get_f())
        pop = algo.evolve(pop)
        diff = olf_pop_f - np.min(pop.get_f())
        pop_weights = pop.get_x()[np.argmin(pop.get_f())]

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("The weight difference is : ", pop_weights[0] * coefficients[0] - weights_to_track[0] * coefficients[0], ' / ',
                                         pop_weights[1] * coefficients[1] - weights_to_track[1] * coefficients[1], ' / ',
                                         pop_weights[2] * coefficients[2] - weights_to_track[2] * coefficients[2], ' / ')

if __name__ == "__main__":
    main()
