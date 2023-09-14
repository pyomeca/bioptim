"""
Test for file IO.
"""

from typing import Callable
from casadi import vertcat, SX, MX
import numpy as np
import pytest
from bioptim import (
    BoundsList,
    ConfigureProblem,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsEvaluation,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    Node,
    NonLinearProgram,
    Solver,
)


class NonControlledMethod:
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self, name: str = None):
        self.a = 0
        self.b = 0
        self.c = 0
        self._name = name

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return NonControlledMethod, dict()

    @property
    def name_dof(self) -> list[str]:
        return ["a", "b", "c"]

    @property
    def nb_state(self):
        return 3

    @property
    def name(self):
        return self._name

    def system_dynamics(
        self,
        a: MX | SX,
        b: MX | SX,
        c: MX | SX,
        t: MX | SX,
        t_phase: MX | SX,
    ) -> MX | SX:
        """
        The system dynamics is the function that describes the model.

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        a_dot = 100 + b
        b_dot = a / (((t - t_phase) + 1) * 100)
        c_dot = a * b + c
        return vertcat(a_dot, b_dot, c_dot)

    def custom_dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        t_phase = nlp.parameters.mx[-1]

        return DynamicsEvaluation(
            dxdt=self.system_dynamics(a=states[0], b=states[1], c=states[2], t=time, t_phase=t_phase),
            defects=None,
        )

    def declare_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
        name = "a"
        name_a = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_a,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        name = "b"
        name_b = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_b,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        name = "c"
        name_c = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_c,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        ConfigureProblem.configure_dynamics_function(ocp, nlp, self.custom_dynamics)


def prepare_ocp(
    n_phase: int,
    time_min: list,
    time_max: list,
    use_sx: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    n_phase: int
        Number of phase
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    ode_solver: OdeSolverBase
        The ode solver to use
    use_sx: bool
        Callable Mx or Sx used for ocp
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    custom_model = NonControlledMethod()
    models = (
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
    )
    n_shooting = [5 for i in range(n_phase)]  # Gives m node shooting for my n phases problem
    final_time = [0.01 for i in range(n_phase)]  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_phase):
        dynamics.add(
            custom_model.declare_variables,
            dynamic_function=custom_model.custom_dynamics,
            phase=i,
            expand=True,
        )

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(n_phase):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, target=5, key="c", node=Node.END, quadratic=True, weight=1, phase=9
    )

    # Sets the bound for all the phases
    x_bounds = BoundsList()
    for i in range(n_phase):
        x_bounds.add("a", min_bound=[[0, 0, 0]], max_bound=[[0 if i == 0 else 1000, 1000, 1000]], phase=i)
        x_bounds.add("b", min_bound=[[0, 0, 0]], max_bound=[[0 if i == 0 else 1000, 1000, 1000]], phase=i)
        x_bounds.add("c", min_bound=[[0, 0, 0]], max_bound=[[0 if i == 0 else 1000, 1000, 1000]], phase=i)

    return OptimalControlProgram(
        models,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        control_type=ControlType.NONE,
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [False])
@pytest.mark.parametrize("use_sx", [False, True])
def test_main_control_type_none(use_sx, assume_phase_dynamics):
    """
    Prepare and solve and animate a reaching task ocp
    """

    # number of stimulation corresponding to phases
    n = 10
    # minimum time between two phase
    time_min = [0.01 for _ in range(n)]
    # maximum time between two phase
    time_max = [0.1 for _ in range(n)]
    ocp = prepare_ocp(
        n_phase=n,
        time_min=time_min,
        time_max=time_max,
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 3.7538907296826786e-14, decimal=15)

    # Check finishing time
    np.testing.assert_almost_equal(sol.time[-1][-1], 0.8194142280853297, decimal=8)

    # Check constraints
    g = np.array(sol.constraints)
    for i in range(n):
        np.testing.assert_almost_equal(g[i * 19 + 0 : i * 19 + 15], np.zeros((15, 1)))
    np.testing.assert_almost_equal(
        g[18:-1:19, 0],
        [0.09839884, 0.09722443, 0.09613614, 0.09451057, 0.09237128, 0.08906275, 0.08277503, 0.06801889, 0.04278909],
    )
    np.testing.assert_equal(g.shape, (187, 1))

    # Check some results
    # first phase
    np.testing.assert_almost_equal(
        sol.states[0]["a"][0], np.array([0.0, 1.96797811, 3.93596411, 5.90396563, 7.87198997, 9.84004415]), decimal=8
    )
    np.testing.assert_almost_equal(
        sol.states[0]["b"][0], np.array([0.0, 0.00020278, 0.00080017, 0.00177652, 0.00311706, 0.00480788]), decimal=8
    )
    np.testing.assert_almost_equal(
        sol.states[0]["c"][0],
        np.array([0.00000000e00, 1.97655504e-06, 3.14066448e-05, 1.57927665e-04, 4.95862761e-04, 1.20288584e-03]),
        decimal=8,
    )

    # intermediate phase
    np.testing.assert_almost_equal(
        sol.states[5]["a"][0],
        np.array([47.87972855, 49.66266983, 51.44572066, 53.22888357, 55.01216107, 56.79555557]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["b"][0],
        np.array([0.09163684, 0.09771425, 0.10393632, 0.11029953, 0.11680052, 0.12343603]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["c"][0],
        np.array([0.61222331, 0.70620465, 0.81049579, 0.92581076, 1.05288683, 1.1924847]),
        decimal=8,
    )

    # last phase
    np.testing.assert_almost_equal(
        sol.states[9]["a"][0],
        np.array([76.18527156, 77.35021111, 78.51521162, 79.6802736, 80.84539753, 82.01058392]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["b"][0],
        np.array([0.2034239, 0.20864607, 0.21391167, 0.21922012, 0.22457085, 0.22996331]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["c"][0],
        np.array([3.75034091, 3.97914086, 4.21820146, 4.46782563, 4.72832104, 5.00000019]),
        decimal=8,
    )
