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
    PhaseDynamics,
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
        b_dot = a / ((t - t_phase) + 1)
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
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

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
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
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
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("use_sx", [False, True])
def test_main_control_type_none(use_sx, phase_dynamics):
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
        phase_dynamics=phase_dynamics,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1.0546674423227002e-12)

    # Check constraints
    g = np.array(sol.constraints)
    for i in range(n):
        np.testing.assert_almost_equal(g[i * 19 + 0 : i * 19 + 15], np.zeros((15, 1)))
    np.testing.assert_almost_equal(
        g[18:-1:19, 0],
        [0.09652524, 0.05752794, 0.0166813, 0.01370305, 0.01262233, 0.01206028, 0.01171445, 0.01147956, 0.01130926],
    )
    np.testing.assert_equal(g.shape, (187, 1))

    # Check some results
    # first phase
    np.testing.assert_almost_equal(
        sol.states[0]["a"][0], np.array([0.0, 1.93089445, 3.86250911, 5.79554351, 7.73067582, 9.66856407]), decimal=2
    )
    np.testing.assert_almost_equal(
        sol.states[0]["b"][0], np.array([0.0, 0.01885085, 0.07450486, 0.16582235, 0.2917295, 0.45121394]), decimal=2
    )
    np.testing.assert_almost_equal(
        sol.states[0]["c"][0], np.array([0.0, 0.00017626, 0.00280773, 0.01415365, 0.044552, 0.10835496]), decimal=2
    )

    # intermediate phase
    np.testing.assert_almost_equal(
        sol.states[5]["a"][0],
        np.array([19.82368302, 20.06913812, 20.31469147, 20.5603441, 20.80609698, 21.05195111]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["b"][0],
        np.array([1.74152087, 1.78205032, 1.82299542, 1.86435469, 1.90612665, 1.94830984]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["c"][0],
        np.array([1.80803497, 1.89726612, 1.98974375, 2.08554493, 2.1847476, 2.28743059]),
        decimal=2,
    )

    # last phase
    np.testing.assert_almost_equal(
        sol.states[9]["a"][0],
        np.array([24.58043802, 24.80998012, 25.03962276, 25.26936668, 25.49921265, 25.72916138]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["b"][0],
        np.array([2.59589214, 2.64067279, 2.68578796, 2.73123661, 2.77701775, 2.82313034]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["c"][0],
        np.array([4.18639942, 4.3405686, 4.49893797, 4.66158258, 4.82857814, 5.00000103]),
        decimal=2,
    )
