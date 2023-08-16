"""
Test for file IO.
"""

from typing import Callable
from casadi import vertcat, SX, MX, Function, sum1, horzcat
import numpy as np
import pytest
from bioptim import (
    BoundsList,
    ConfigureProblem,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    Node,
    NonLinearProgram,
    Solver,
    InitialGuessList,
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
    def name_dof(self)->list[str]:
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
        t_phase = nlp.parameters.cx[-1]

        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(a=states[0], b=states[1], c=states[2], t=time, t_phase=t_phase),
            defects=None,
        )

    # def custom_configure_dynamics_function(self, ocp, nlp, **extra_params):
    #     """
    #     Configure the dynamics of the system
    #     """
    #
    #     nlp.parameters = ocp.parameters
    #     DynamicsFunctions.apply_parameters(nlp.parameters.cx, nlp)
    #
    #     dynamics_eval = self.custom_dynamics(
    #         nlp.t0, nlp.states.scaled.cx, nlp.controls.scaled.cx, nlp.parameters.cx, nlp, **extra_params
    #     )
    #
    #     dynamics_dxdt = dynamics_eval.dxdt
    #     if isinstance(dynamics_dxdt, (list, tuple)):
    #         dynamics_dxdt = vertcat(*dynamics_dxdt)
    #
    #     nlp.dynamics_func = Function(
    #         "ForwardDyn",
    #         [
    #             nlp.time.mx,
    #             nlp.states.scaled.mx_reduced,
    #             nlp.controls.scaled.mx_reduced,
    #             nlp.parameters.mx,
    #             nlp.stochastic_variables.scaled.mx,
    #         ],
    #         [dynamics_dxdt],
    #         ["t", "x", "u", "p", "s"],
    #         ["xdot"],
    #     )

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

        # name = "t"
        # name_t = [name]
        # ConfigureProblem.configure_new_variable(
        #     name,
        #     name_t,
        #     ocp,
        #     nlp,
        #     as_states=False,
        #     as_controls=False,
        #     as_states_dot=False,
        # )

        ConfigureProblem.configure_t(ocp, nlp, as_states=False, as_controls=False)

        # t = MX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function
        # self.custom_configure_dynamics_function(ocp, nlp, t=t)
        # self.custom_configure_dynamics_function(ocp, nlp)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, self.custom_dynamics)


def prepare_ocp(
    n_phase: int,
    time_min: list,
    time_max: list,
    use_sx: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=1),
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
    models = [custom_model for i in range(n_phase)]  # Gives custom_model as model for n phases
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

    # x_init = InitialGuessList()
    # variable_bound_list = NonControlledMethod().name_dof
    # for i in range(n_phase):
    #     for j in range(len(variable_bound_list)):
    #         x_init.add(variable_bound_list[j], [0])
    #
    # # Creates the controls of our problem (in our case, equals to an empty list)
    # u_bounds = BoundsList()
    # for i in range(n_phase):
    #     u_bounds.add("", min_bound=[], max_bound=[])
    #
    # u_init = InitialGuessList()
    # for i in range(n_phase):
    #     u_init.add("", min_bound=[], max_bound=[])

    return OptimalControlProgram(
        models,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        # x_init=x_init,
        # u_init=u_init,
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

    # TODO It seems assume_phase_dynamics=True is broken
    #  I THINK IT'S NORMAL AS THIS FUN CHANGES AT EACH NODE

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
    np.testing.assert_almost_equal(f[0, 0], 1.0546674423227002e-12)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (187, 1))
    np.testing.assert_almost_equal(
        g,
        np.array(
            [
                [8.88178420e-16],
                [-3.46944695e-18],
                [-3.79470760e-19],
                [0.00000000e00],
                [-2.77555756e-17],
                [-5.63785130e-18],
                [0.00000000e00],
                [0.00000000e00],
                [-2.08166817e-17],
                [0.00000000e00],
                [-1.11022302e-16],
                [-6.24500451e-17],
                [1.77635684e-15],
                [-1.11022302e-16],
                [-1.52655666e-16],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [9.65386567e-02],
                [-1.06581410e-14],
                [-8.99280650e-14],
                [-1.65839564e-13],
                [-2.84217094e-14],
                [-2.36255460e-13],
                [-5.93303184e-13],
                [-5.32907052e-14],
                [-3.75810494e-13],
                [-1.21935795e-12],
                [-7.99360578e-14],
                [-5.08593168e-13],
                [-2.07489581e-12],
                [-1.11910481e-13],
                [-6.35047570e-13],
                [-3.19000382e-12],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [5.69680479e-02],
                [-1.06581410e-14],
                [-4.55191440e-14],
                [-4.68514116e-13],
                [-1.06581410e-14],
                [-4.61852778e-14],
                [-4.94271291e-13],
                [-1.06581410e-14],
                [-4.66293670e-14],
                [-5.21027665e-13],
                [-1.06581410e-14],
                [-4.68514116e-14],
                [-5.48672219e-13],
                [-1.42108547e-14],
                [-4.72955008e-14],
                [-5.77093928e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.66599664e-02],
                [-7.10542736e-15],
                [-1.64313008e-14],
                [-3.67039732e-13],
                [-3.55271368e-15],
                [-1.62092562e-14],
                [-3.78141962e-13],
                [-7.10542736e-15],
                [-1.62092562e-14],
                [-3.90132371e-13],
                [-7.10542736e-15],
                [-1.62092562e-14],
                [-4.02122780e-13],
                [-3.55271368e-15],
                [-1.64313008e-14],
                [-4.13891144e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.36970015e-02],
                [-7.10542736e-15],
                [-7.54951657e-15],
                [-3.42170736e-13],
                [0.00000000e00],
                [-7.54951657e-15],
                [-3.49942297e-13],
                [-7.10542736e-15],
                [-7.32747196e-15],
                [-3.58157948e-13],
                [-3.55271368e-15],
                [-7.32747196e-15],
                [-3.66595643e-13],
                [-3.55271368e-15],
                [-7.77156117e-15],
                [-3.74589249e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.26197006e-02],
                [-3.55271368e-15],
                [-3.55271368e-15],
                [-3.37729844e-13],
                [-3.55271368e-15],
                [-3.55271368e-15],
                [-3.44169138e-13],
                [-3.55271368e-15],
                [-3.55271368e-15],
                [-3.50830476e-13],
                [0.00000000e00],
                [-3.55271368e-15],
                [-3.57935903e-13],
                [-3.55271368e-15],
                [-3.55271368e-15],
                [-3.64153152e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.20589659e-02],
                [-3.55271368e-15],
                [-1.33226763e-15],
                [-3.41504602e-13],
                [-3.55271368e-15],
                [-8.88178420e-16],
                [-3.46389584e-13],
                [-3.55271368e-15],
                [-1.33226763e-15],
                [-3.52162743e-13],
                [-3.55271368e-15],
                [-1.33226763e-15],
                [-3.58379992e-13],
                [-3.55271368e-15],
                [-8.88178420e-16],
                [-3.64153152e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.17137752e-02],
                [-3.55271368e-15],
                [0.00000000e00],
                [-3.48165941e-13],
                [-3.55271368e-15],
                [4.44089210e-16],
                [-3.53495011e-13],
                [-3.55271368e-15],
                [4.44089210e-16],
                [-3.58379992e-13],
                [-3.55271368e-15],
                [0.00000000e00],
                [-3.63709063e-13],
                [-3.55271368e-15],
                [0.00000000e00],
                [-3.69038133e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.14792488e-02],
                [0.00000000e00],
                [1.33226763e-15],
                [-3.57047725e-13],
                [-3.55271368e-15],
                [1.33226763e-15],
                [-3.61932706e-13],
                [0.00000000e00],
                [8.88178420e-16],
                [-3.66817687e-13],
                [0.00000000e00],
                [8.88178420e-16],
                [-3.71258579e-13],
                [0.00000000e00],
                [1.33226763e-15],
                [-3.76587650e-13],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.13091730e-02],
                [-3.55271368e-15],
                [2.22044605e-15],
                [-3.70370401e-13],
                [-3.55271368e-15],
                [1.33226763e-15],
                [-3.74811293e-13],
                [0.00000000e00],
                [2.22044605e-15],
                [-3.78364007e-13],
                [0.00000000e00],
                [1.77635684e-15],
                [-3.82804899e-13],
                [0.00000000e00],
                [1.33226763e-15],
                [-3.88133969e-13],
                [1.11844907e-02],
            ]
        ),
    )

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
        np.array([19.76647875, 20.01192073, 20.25746084, 20.50310011, 20.74883957, 20.99468026]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["b"][0],
        np.array([1.74723366, 1.78770329, 1.8286031, 1.86993181, 1.91168818, 1.95387098]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[5]["c"][0],
        np.array([1.80854721, 1.89779906, 1.9902922, 2.08610429, 2.18531387, 2.28800045]),
        decimal=2,
    )

    # last phase
    np.testing.assert_almost_equal(
        sol.states[9]["a"][0],
        np.array([24.52324217, 24.75280044, 24.98245916, 25.21221914, 25.44208114, 25.67204593]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["b"][0],
        np.array([2.60114035, 2.64587979, 2.69096794, 2.73640393, 2.78218689, 2.82831598]),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        sol.states[9]["c"][0],
        np.array([4.18665929, 4.34078482, 4.49910564, 4.66169752, 4.82863693, 5.00000103]),
        decimal=2,
    )
