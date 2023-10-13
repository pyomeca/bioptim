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

    def __init__(self, name: str = None, from_time: bool = False):
        self.a = 0
        self.b = 0
        self.c = 0
        self._name = name
        self._from_time = from_time

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

    @staticmethod
    def custom_dynamics_from_param(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        t_phase = nlp.parameters.mx[-1]
        print("t_phase", t_phase)
        print("t_parameter", nlp.parameters.mx[-1])
        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(a=states[0], b=states[1], c=states[2], t=time, t_phase=t_phase),
            defects=None,
        )

    @staticmethod
    def custom_dynamics_from_time(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        t_phase: MX | SX,
    ) -> DynamicsEvaluation:
        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(a=states[0], b=states[1], c=states[2], t=time, t_phase=t_phase),
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

        if self._from_time:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                self.custom_dynamics_from_time,
                t_phase=ocp.node_time(phase_idx=nlp.phase_idx, node_idx=0, type="mx"),
            )
        else:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, self.custom_dynamics_from_param)


def prepare_ocp(
    n_phase: int,
    time_min: list,
    time_max: list,
    use_sx: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    from_time: bool = False,
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
    from_time: bool
        If the dynamics function should be called with the time or with the t_phase parameter

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    custom_model = NonControlledMethod(from_time=from_time)
    models = (
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
        NonControlledMethod(from_time=from_time),
    )
    n_shooting = [5 for i in range(n_phase)]  # Gives m node shooting for my n phases problem
    final_time = [0.01 for i in range(n_phase)]  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_phase):
        dynamics.add(
            custom_model.declare_variables,
            dynamic_function=custom_model.custom_dynamics_from_time
            if from_time
            else custom_model.custom_dynamics_from_param,
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
@pytest.mark.parametrize("from_time", [False, True])
def test_main_control_type_none(use_sx, phase_dynamics, from_time):
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
        from_time=from_time,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    if from_time:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.5070739287980432)

        # Check finishing time
        np.testing.assert_almost_equal(sol.time[-1][-1], 0.7755731313184815)

        # Check constraints
        g = np.array(sol.constraints)
        for i in range(n):
            np.testing.assert_almost_equal(g[i * 19 + 0 : i * 19 + 15], np.zeros((15, 1)))
        np.testing.assert_almost_equal(
            g[18:-1:19, 0],
            [
                0.09999289,
                0.09999639,
                0.09999735,
                0.09999766,
                0.09999762,
                0.09999711,
                0.09999512,
                0.05555808,
                0.01002101,
            ],
        )
        np.testing.assert_equal(g.shape, (187, 1))

        # Check some results
        # first phase
        np.testing.assert_almost_equal(
            sol.states[0]["a"][0],
            np.array([0.0, 1.99985916, 3.99972614, 5.99960848, 7.99951343, 9.99944799]),
            decimal=8,
        )
        np.testing.assert_almost_equal(
            sol.states[0]["b"][0],
            np.array([0.0, 0.00019734, 0.00077918, 0.00173086, 0.00303855, 0.00468921]),
            decimal=8,
        )
        np.testing.assert_almost_equal(
            sol.states[0]["c"][0],
            np.array([0.00000000e00, 1.98632308e-06, 3.15783091e-05, 1.58871623e-04, 4.99074209e-04, 1.21126322e-03]),
            decimal=8,
        )

        # intermediate phase
        np.testing.assert_almost_equal(
            sol.states[5]["a"][0],
            np.array([50.01804756, 52.02046588, 54.02308818, 56.02591824, 58.02895969, 60.03221604]),
            decimal=8,
        )
        np.testing.assert_almost_equal(
            sol.states[5]["b"][0],
            np.array([0.11877436, 0.12887657, 0.13917146, 0.14965174, 0.16031051, 0.17114126]),
            decimal=8,
        )
        np.testing.assert_almost_equal(
            sol.states[5]["c"][0],
            np.array([0.82592097, 0.97021453, 1.13335438, 1.316757, 1.52188243, 1.75023443]),
            decimal=8,
        )

        # last phase
        np.testing.assert_almost_equal(
            sol.states[9]["a"][0],
            np.array([76.62662064, 76.82758133, 77.02854511, 77.22951197, 77.43048192, 77.63145494]),
            decimal=8,
        )
        np.testing.assert_almost_equal(
            sol.states[9]["b"][0],
            np.array([0.28001304, 0.28154909, 0.2830861, 0.28462404, 0.28616293, 0.28770275]),
            decimal=8,
        )
        np.testing.assert_almost_equal(
            sol.states[9]["c"][0],
            np.array([4.813047, 4.86591798, 4.91924532, 4.97303132, 5.02727829, 5.08198854]),
            decimal=8,
        )

    else:
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
            sol.states[0]["a"][0],
            np.array([0.0, 1.96960231, 3.93921216, 5.90883684, 7.87848335, 9.84815843]),
            decimal=8,
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
