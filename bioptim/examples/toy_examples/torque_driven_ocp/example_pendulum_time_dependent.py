"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This example is a good place to start investigating bioptim's time dependent problems as it describes a time dependent
torque driven dynamic, it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

import platform

import numpy as np

from casadi import MX, SX, sin, vertcat

from bioptim import (
    TorqueBiorbdModel,
    BoundsList,
    ControlType,
    CostType,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsOptions,
    InitialGuessList,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    NonLinearProgram,
    Solver,
    PhaseDynamics,
)
from bioptim.examples.utils import ExampleUtils


class TimeDependentModel(TorqueBiorbdModel):
    def dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

        Parameters
        ----------
        time: MX | SX
            The time of the system
        states: MX | SX
            The state of the system
        controls: MX | SX
            The controls of the system
        parameters: MX | SX
            The parameters acting on the system
        algebraic_states: MX | SX
            The algebraic states variables of the system
        numerical_timeseries: MX | SX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase

        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (sin(time) * time.ones(nlp.model.nb_tau) * 10)

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = self.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    phase_dynamics: PhaseDynamics = PhaseDynamics.ONE_PER_NODE,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TimeDependentModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    expand = not isinstance(ode_solver, OdeSolver.IRK)
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
        expand_dynamics=expand,
        phase_dynamics=phase_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0  # Start and end at 0...
    x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Define control path constraint
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        control_type=control_type,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    biorbd_model_path = ExampleUtils.folder + "/models/pendulum.bioMod"

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, final_time=1, n_shooting=30)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol.graphs(show_bounds=True)

    # --- Show the results in a bioviz animation --- #
    sol.print_cost()
    # sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
