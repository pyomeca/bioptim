import numpy as np

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    BiorbdModel,
    DynamicsFcn,
    RigidBodyDynamics,
)


def prepare_ocp(biorbd_model_path: str, use_sx: bool = False, ode_solver=OdeSolver.RK4(), rigidbody_dynamics=RigidBodyDynamics.ODE, with_ligament=False) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    use_sx: bool
        If the project should be build in mx [False] or sx [True]
    ode_solver: OdeSolver
        The type of integrator
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    # Model path
    bio_model = BiorbdModel(biorbd_model_path)

    # ConfigureProblem parameters
    number_shooting_points = 100
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10000000)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN,rigidbody_dynamics=rigidbody_dynamics, with_ligament=with_ligament)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[0, 0] = 0
    x_bounds[1, 0] = 0
    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * bio_model.nb_tau ,
        [tau_max] * bio_model.nb_tau ,
    )
    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0]*bio_model.nb_q + [0]*bio_model.nb_qdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        n_threads=8,
        use_sx=use_sx,
    )

def main():


    model_path = "/home/lim/Documents/Stage_Antoine/Antoine_Leroy/Optimization/bioptim/bioptim/examples/torque_driven_ocp/models/Mass_with_ligament.bioMod"

    ocp = prepare_ocp(biorbd_model_path=model_path, with_ligament=True)

    # --- Solve the program --- #

    sol = ocp.solve()

    sol.graphs()


if __name__ == "__main__":

    # import bioviz
    # b = bioviz.Viz(model_path="pendulum_with_ligament.bioMod", show_floor=False, show_muscles=True)
    # b.exec()

    main()

