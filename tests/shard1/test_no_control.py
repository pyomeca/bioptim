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
def test_no_control(use_sx, phase_dynamics):
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

    # Check integration results

    states = (0, 0, 0)
    controls = 0
    parameters = 0.5  # time parameter
    stochastics = 0

    phase0_node0_dynamics = ocp.nlp[0].dynamics[0]
    phase0_node2_dynamics = ocp.nlp[0].dynamics[2]
    phase0_node4_dynamics = ocp.nlp[0].dynamics[4]
    phase4_node0_dynamics = ocp.nlp[4].dynamics[0]
    phase4_node2_dynamics = ocp.nlp[4].dynamics[2]
    phase4_node4_dynamics = ocp.nlp[4].dynamics[4]
    phase9_node0_dynamics = ocp.nlp[9].dynamics[0]
    phase9_node2_dynamics = ocp.nlp[9].dynamics[2]
    phase9_node4_dynamics = ocp.nlp[9].dynamics[4]

    # first phase, first node
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[10.0003, 0.00883935, 0.00230952]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 2, 4.00002, 6.00007, 8.00016, 10.0003],
                    [0, 0.000389643, 0.00151948, 0.00333568, 0.00579005, 0.00883935],
                    [0, 3.93465e-06, 6.19004e-05, 0.00030844, 0.00096004, 0.00230952],
                ]
            )
        ),
        decimal=4,
    )

    # first phase, intermediate node

    states = (4, 0, 0)
    np.testing.assert_almost_equal(
        phase0_node2_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[14.0005, 0.0118695, 0.00559404]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase0_node2_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [4, 6.00001, 8.00006, 10.0001, 12.0003, 14.0005],
                    [0, 0.00140722, 0.00332391, 0.00572289, 0.00857913, 0.0118695],
                    [0, 7.11181e-05, 0.000407076, 0.00123765, 0.00285158, 0.00559404],
                ]
            )
        ),
        decimal=4,
    )

    # first phase, last node
    states = (0, 3, 0)
    np.testing.assert_almost_equal(
        phase0_node4_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[10.3002, 3.00533, 1.59925]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase0_node4_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 2.06, 4.12001, 6.18004, 8.24009, 10.3002],
                    [3, 3.00023, 3.00089, 3.00197, 3.00346, 3.00533],
                    [0, 0.0622164, 0.250567, 0.567681, 1.0163, 1.59925],
                ]
            )
        ),
        decimal=4,
    )

    # intermediate phase, first node
    states = (0, 0, 5)
    np.testing.assert_almost_equal(
        phase4_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[10.0001, 0.00194822, 5.52635]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase4_node0_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 2, 4, 6.00001, 8.00003, 10.0001],
                    [0, 7.95759e-05, 0.000316627, 0.000708684, 0.00125333, 0.00194822],
                    [5, 5.10101, 5.20407, 5.30925, 5.41664, 5.52635],
                ]
            )
        ),
        decimal=4,
    )

    # intermediate phase, intermediate node
    states = (0, 0, 0)
    controls = 2
    np.testing.assert_almost_equal(
        phase4_node2_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[10.0001, 0.00180737, 0.00046329]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase4_node2_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 2, 4, 6.00001, 8.00003, 10.0001],
                    [0, 7.37103e-05, 0.000293402, 0.000656953, 0.00116228, 0.00180737],
                    [0, 7.40468e-07, 1.18532e-05, 6.00179e-05, 0.000189722, 0.00046329],
                ]
            )
        ),
        decimal=4,
    )

    # intermediate phase, last node
    controls = 0
    parameters = 1
    np.testing.assert_almost_equal(
        phase4_node4_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[20.0003, 0.00405445, 0.00424571]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase4_node4_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 4, 8.00002, 12.0001, 16.0001, 20.0003],
                    [0, 0.000165747, 0.000659351, 0.00147546, 0.00260886, 0.00405445],
                    [0, 6.68327e-06, 0.00010743, 0.000545968, 0.00173221, 0.00424571],
                ]
            )
        ),
        decimal=4,
    )

    # last phase, first node
    parameters = 0.5
    stochastics = 8
    np.testing.assert_almost_equal(
        phase9_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[10, 0.000986865, 0.000252406]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase9_node0_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 2, 4, 6.00001, 8.00002, 10],
                    [0, 3.98937e-05, 0.000159152, 0.000357146, 0.000633255, 0.000986865],
                    [0, 4.00532e-07, 6.42375e-06, 3.25842e-05, 0.000103183, 0.000252406],
                ]
            )
        ),
        decimal=4,
    )

    # last phase, intermediate node
    states = (3, 20, 10)
    stochastics = 0
    np.testing.assert_almost_equal(
        phase9_node2_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[15.0001, 20.0017, 29.773]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase9_node2_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [3, 5.4, 7.80001, 10.2, 12.6, 15.0001],
                    [20, 20.0002, 20.0004, 20.0008, 20.0012, 20.0017],
                    [10, 11.8973, 14.8027, 18.7364, 23.7195, 29.773],
                ]
            )
        ),
        decimal=4,
    )

    # last phase, last node
    states = (3, 6, 8)
    controls = 2
    parameters = 1.5
    stochastics = 3
    np.testing.assert_almost_equal(
        phase9_node4_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[34.8004, 6.00394, 48.8187]]),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        phase9_node4_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [3, 9.36001, 15.72, 22.0801, 28.4402, 34.8004],
                    [6, 6.00026, 6.00079, 6.00158, 6.00263, 6.00394],
                    [8, 10.7758, 16.0834, 24.0798, 34.9323, 48.8187],
                ]
            )
        ),
        decimal=4,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # Check optimization results
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
