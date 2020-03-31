import biorbd
from matplotlib import pyplot as plt

import casadi_nlp
from casadi_nlp.objective_functions import ObjectiveFunction
from casadi_nlp.constraints import Constraint
from casadi_nlp.dynamics import Dynamics

# --- Options --- #
# Model path
biorbd_model = biorbd.Model("eocar.bioMod")

# Results path
optimization_name = "eocarBiorbd"
results_path = "Results/"
control_results_file_name = results_path + "Controls" + optimization_name + ".txt"
state_results_file_name = results_path + "States" + optimization_name + ".txt"

# Problem parameters
number_shooting_points = 30
final_time = 2
ode_solver = casadi_nlp.OdeSolver.RK
velocity_max = 15
is_cyclic_constraint = False
is_cyclic_objective = False

# Add objective functions
objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

# Dynamics
variable_type = casadi_nlp.Variable.variable_torque_driven
dynamics_func = Dynamics.forward_dynamics_torque_driven

# Constraints
constraints = ((Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
               (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),)

# Define path constraint
X_bounds = casadi_nlp.Bounds()
X_init = casadi_nlp.InitialConditions()
ranges = []
for i in range(biorbd_model.nbSegment()):
    segRanges = biorbd_model.segment(i).ranges()
    for j in range(len(segRanges)):
        ranges.append(biorbd_model.segment(i).ranges()[j])

for i in range(biorbd_model.nbQ()):
    if i == 3:
        X_bounds.first_node_min.append(0)
        X_bounds.first_node_max.append(0)
    else:
        X_bounds.first_node_min.append(ranges[i].min())
        X_bounds.first_node_max.append(ranges[i].max())
    X_bounds.min.append(ranges[i].min())
    X_bounds.max.append(ranges[i].max())
    if i == 3:
        X_bounds.last_node_min.append(1.57)
        X_bounds.last_node_max.append(1.57)
    else:
        X_bounds.last_node_min.append(ranges[i].min())
        X_bounds.last_node_max.append(ranges[i].max())
    X_init.init.append(0)
for i in range(biorbd_model.nbQdot()):
    X_bounds.first_node_min.append(0)
    X_bounds.first_node_max.append(0)
    X_bounds.min.append(-velocity_max)
    X_bounds.max.append(velocity_max)
    X_bounds.last_node_min.append(0)
    X_bounds.last_node_max.append(0)
    X_init.init.append(0)

U_bounds = casadi_nlp.Bounds()
U_init = casadi_nlp.InitialConditions()
for i in range(biorbd_model.nbGeneralizedTorque()):
    U_bounds.min.append(-100)
    U_bounds.max.append(100)
    U_init.init.append(0)
# ------------- #

# --- Solve the program --- #
nlp = casadi_nlp.OptimalControlProgram(
    biorbd_model, variable_type, dynamics_func, ode_solver, number_shooting_points, final_time,
    objective_functions, X_init, U_init, X_bounds, U_bounds,
    constraints, is_cyclic_constraint=is_cyclic_constraint, is_cyclic_objective=is_cyclic_objective)

sol = nlp.solve()
for idx in range(biorbd_model.nbQ()):
    plt.figure()
    q = sol["x"][0*biorbd_model.nbQ()+idx::3*biorbd_model.nbQ()]
    q_dot = sol["x"][1*biorbd_model.nbQ()+idx::3*biorbd_model.nbQ()]
    u = sol["x"][2*biorbd_model.nbQ()+idx::3*biorbd_model.nbQ()]
    plt.plot(q)
    plt.plot(q_dot)
    plt.plot(u)
plt.show()
