import sys

import biorbd
from matplotlib import pyplot as plt

import biorbd_optim
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.dynamics import Dynamics

# --- Options --- #
# Model path
biorbd_model = biorbd.Model("eocar.bioMod")

# Results path
optimization_name = "eocar"
results_path = "Results/"
control_results_file_name = results_path + "Controls" + optimization_name + ".txt"
state_results_file_name = results_path + "States" + optimization_name + ".txt"

# Problem parameters
number_shooting_points = 30
final_time = 2
ode_solver = biorbd_optim.OdeSolver.RK
is_cyclic_constraint = False
is_cyclic_objective = False

# Add objective functions
objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

# Dynamics
variable_type = biorbd_optim.Variable.torque_driven
dynamics_func = Dynamics.forward_dynamics_torque_driven

# Geometric constraints
constraints = ((Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
               (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),)

# Path constraint
X_bounds = biorbd_optim.Bounds()
X_init = biorbd_optim.InitialConditions()

# First node states bounds
X_bounds.first_node_min = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
X_bounds.first_node_max = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())

# Last node states bounds
X_bounds.last_node_min = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())
X_bounds.last_node_min[3] = 1
X_bounds.first_node_max = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())

# Gets bounds from biorbd model
ranges = []
for i in range(biorbd_model.nbSegment()):
    ranges.extend([biorbd_model.segment(i).ranges()[j] for j in range(len(biorbd_model.segment(i).ranges()))])
X_bounds.min = [ranges[i].min() for i in range(biorbd_model.nbQ())]
X_bounds.max = [ranges[i].max() for i in range(biorbd_model.nbQ())]

# Path constraint velocity
velocity_max = 15
X_bounds.min.extend([-velocity_max] * (biorbd_model.nbQdot()))
X_bounds.max.extend([velocity_max] * (biorbd_model.nbQdot()))

# Initial guess
X_init.init = [0] * (biorbd_model.nbQ() + biorbd_model.nbQdot())

# Define control path constraint
torque_min = -100
torque_max = 100
torque_init = 0
U_bounds = biorbd_optim.Bounds()
U_init = biorbd_optim.InitialConditions()

U_bounds.min = [torque_min for _ in range(biorbd_model.nbGeneralizedTorque())]
U_bounds.max = [torque_max for _ in range(biorbd_model.nbGeneralizedTorque())]
U_init.init = [torque_init for _ in range(biorbd_model.nbGeneralizedTorque())]
# ------------- #

# --- Solve the program --- #
nlp = biorbd_optim.OptimalControlProgram(
    biorbd_model, variable_type, dynamics_func, ode_solver, number_shooting_points, final_time,
    objective_functions, X_init, U_init, X_bounds, U_bounds,
    constraints, is_cyclic_constraint=is_cyclic_constraint, is_cyclic_objective=is_cyclic_objective)

sol = nlp.solve()
# ------------- #


# --- Plot the solution --- #
for idx in range(biorbd_model.nbQ()):
    plt.figure()
    q = sol["x"][0*biorbd_model.nbQ()+idx::3*biorbd_model.nbQ()]
    q_dot = sol["x"][1*biorbd_model.nbQ()+idx::3*biorbd_model.nbQ()]
    u = sol["x"][2*biorbd_model.nbQ()+idx::3*biorbd_model.nbQ()]
    plt.plot(q)
    plt.plot(q_dot)
    plt.plot(u)
plt.show()
# ------------- #

