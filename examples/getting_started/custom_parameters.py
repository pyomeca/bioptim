import biorbd

from bioptim import (
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    Objective,
    ObjectiveFcn,
    InterpolationType,
    Data,
    ParameterList,
    OdeSolver,
)


def my_parameter_function(biorbd_model, value, extra_value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which described by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, value * extra_value))


def my_target_function(ocp, value):
    # The target function is a penalty function.
    # `ocp` and `value` are mandatory. The rest is defined in the
    # parameter by the user
    # Please note that if a target value exist (target parameters) it is automatically added, and therefore
    # should not be added by hand here (that is, the next line should not read: return value - target)
    return value


def prepare_ocp(
    biorbd_model_path,
    final_time,
    number_shooting_points,
    min_g,
    max_g,
    target_g,
    ode_solver=OdeSolver.RK4,
    use_SX=False,
):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -30, 30, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=10)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    parameters = ParameterList()
    bound_gravity = Bounds(min_g, max_g, interpolation=InterpolationType.CONSTANT)
    # and an initial condition
    initial_gravity = InitialGuess((min_g + max_g) / 2)
    parameter_objective_functions = Objective(
        my_target_function, weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target=target_g
    )
    parameters.add(
        "gravity_z",  # The name of the parameter
        my_parameter_function,  # The function that modifies the biorbd model
        initial_gravity,  # The initial guess
        bound_gravity,  # The bounds
        size=1,  # The number of elements this particular parameter vector has
        penalty_list=parameter_objective_functions,  # ObjectiveFcn of constraint for this particular parameter
        extra_value=1,  # You can define as many extra arguments as you want
    )

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
        parameters=parameters,
        ode_solver=ode_solver,
        use_SX=use_SX,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, min_g=-10, max_g=-6, target_g=-8
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    length = params["gravity_z"][0, 0]
    print(length)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=200)
