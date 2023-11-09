"""
Test for file IO.
"""

from typing import Callable

import matplotlib.pyplot as plt
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
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    Node,
    NonLinearProgram,
    Shooting,
    Solution,
    SolutionIntegrator,
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

    def initial_values(self) -> np.array:
        """
        Returns
        -------
        The initial values of a, b, c
        """
        return np.array([[self.a], [self.b], [self.c]])

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
    use_sx: bool
        Callable Mx or Sx used for ocp
    ode_solver: OdeSolverBase
        The ode solver to use
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

    # Creates the objective for my last phases
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, target=5, key="c", node=Node.END, quadratic=True, weight=1, phase=9
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, target=100, key="a", node=Node.END, quadratic=True, weight=0.001, phase=9
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

    # number of phases
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
    np.testing.assert_almost_equal(f[0, 0], 0.2919065990591678)

    # Check finishing time
    np.testing.assert_almost_equal(sol.time[-1][-1], 0.8299336018055604)

    # Check constraints
    g = np.array(sol.constraints)
    for i in range(n):
        np.testing.assert_almost_equal(g[i * 19 + 0 : i * 19 + 15], np.zeros((15, 1)))
    np.testing.assert_almost_equal(
        g[18:-1:19, 0],
        [0.09848005, 0.0974753, 0.09652673, 0.09540809, 0.0939693, 0.09197322, 0.08894771, 0.08377719, 0.07337567],
    )
    np.testing.assert_equal(g.shape, (187, 1))

    # Check some results
    # first phase
    np.testing.assert_almost_equal(
        sol.states[0]["a"][0], np.array([0.0, 1.96960231, 3.93921216, 5.90883684, 7.87848335, 9.84815843]), decimal=8
    )
    np.testing.assert_almost_equal(
        sol.states[0]["b"][0], np.array([0.0, 0.00019337, 0.00076352, 0.00169617, 0.00297785, 0.0045958]), decimal=8
    )
    np.testing.assert_almost_equal(
        sol.states[0]["c"][0],
        np.array([0.00000000e00, 1.88768128e-06, 3.00098595e-05, 1.50979104e-04, 4.74274962e-04, 1.15105831e-03]),
        decimal=8,
    )

    # intermediate phase
    np.testing.assert_almost_equal(
        sol.states[5]["a"][0],
        np.array([48.20121535, 50.04237763, 51.88365353, 53.72504579, 55.56655709, 57.40819004]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["b"][0],
        np.array([0.08926236, 0.0953631, 0.10161488, 0.10801404, 0.11455708, 0.1212406]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["c"][0],
        np.array([0.60374532, 0.69912979, 0.80528341, 0.92297482, 1.05299864, 1.19617563]),
        decimal=8,
    )

    # last phase
    np.testing.assert_almost_equal(
        sol.states[9]["a"][0],
        np.array([82.06013653, 82.2605896, 82.4610445, 82.6615012, 82.86195973, 83.06242009]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["b"][0],
        np.array([0.22271563, 0.22362304, 0.22453167, 0.2254415, 0.22635253, 0.22726477]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["c"][0],
        np.array([4.83559727, 4.88198772, 4.92871034, 4.97576671, 5.02315844, 5.07088713]),
        decimal=8,
    )


@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
@pytest.mark.parametrize("use_sx", [False, True])
def test_integration_control_type_none(integrator, use_sx):
    """
    Prepare and integrate the ocp
    """

    # Number of phases
    n = 10

    # For this test, minimum and maximum time between two phase won't affect the dynamics
    # as the phase_time is set below and sent into parameters initial guess.
    # But required to create the ocp.
    time_min = [0.01 for _ in range(n)]
    time_max = [0.1 for _ in range(n)]
    phase_time = [0.01 * _ for _ in range(1, n + 1)]

    # Create the ocp
    ocp = prepare_ocp(
        n_phase=n,
        time_min=time_min,
        time_max=time_max,
        use_sx=use_sx,
    )

    # Initialize the initial guesses
    x = InitialGuessList()
    u = InitialGuessList()
    p = InitialGuessList()
    s = InitialGuessList()

    for i in range(len(ocp.nlp)):
        for j in range(len(ocp.nlp[0].states.keys())):
            x.add(ocp.nlp[0].states.keys()[j], ocp.nlp[0].model.initial_values()[j], phase=i)
        if len(ocp.parameters) != 0:
            for k in range(len(ocp.parameters)):
                p.add(ocp.parameters.keys()[k], initial_guess=np.array([phase_time[k]]), phase=i)
        # TODO: Once we allow the user to send empty dictionaries to InitialGuessList, we can remove these lines
        # else:
            # p.add(key="", phase=i)
        # u.add(key="", phase=i)
        # s.add(key="", phase=i)

    sol_from_initial_guess = Solution.from_initial_guess(ocp, [x, u, p, s])

    result = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, merge_phases=True, integrator=integrator)
    # Merging from integration and outside integration process is not the same, therefor we need to test it
    result = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, merge_phases=False, integrator=integrator)
    result = result.merge_phases()

    # Check if continuity is respected
    np.testing.assert_almost_equal(
        result.states["c"][-1][-1],
        0.0012279529390333628,
    )

    # Keeping those lines for debugging purposes, REMOVE THEM WHEN MERGE WITH MAIN
    # plt.plot(result.time, result.states["c"][0])
    # plt.show()
