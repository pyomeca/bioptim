"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""
from typing import Callable
import numpy as np
from casadi import MX, exp, vertcat


class DingModel:
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self, name: str = None):
        self._name = name
        # custom values for the example
        self.tauc = 0.020  # Value from Ding's experimentation [1] (s)
        self.r0_km_relationship = 1.04  # (unitless)
        # Different values for each person :
        self.alpha_a = -4.0 * 10e-7  # Value from Ding's experimentation [1] (s^-2)
        self.alpha_tau1 = 2.1 * 10e-5  # Value from Ding's experimentation [1] (N^-1)
        self.tau2 = 0.060  # Close value from Ding's experimentation [2] (s)
        self.tau_fat = 127  # Value from Ding's experimentation [1] (s)
        self.alpha_km = 1.9 * 10e-8  # Value from Ding's experimentation [1] (s^-1.N^-1)
        self.a_rest = 3009  # Value from Ding's experimentation [1] (N.s-1)
        self.tau1_rest = 0.050957  # Value from Ding's experimentation [1] (s)
        self.km_rest = 0.103  # Value from Ding's experimentation [1] (unitless)

        # TODO : Fix the error at the beginning of the F curve
        # Works better with RK4 or RK8, COLLOCATION methode doesn't work
        # TODO : Constrain nodes for constant stimulation interval
        # Where to code, in prepare ocp, create a new constrain func ?
        # TODO : Add intensity as parameter to the model

    def standard_rest_values(self) -> np.array:
        """
        Returns
        -------
        The rested values of Cn, F, A, Tau1, Km
        """
        return np.array([[0], [0], [self.a_rest], [self.tau1_rest], [self.km_rest]])

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return DingModel, dict()  # todo : pas compris comment remplir le dict

    # essai de dict : dict(("tauc", self.tauc), ("a_rest", self.a_rest), ("tau1_rest", self.tau1_rest),
    #                      ("km_rest", self.km_rest), ("tau2", self.tau2), ("alpha_a", self.alpha_a),
    #                      ("alpha_tau1", self.alpha_tau1),("alpha_km", self.alpha_km),("tau_fat", self.tau_fat))

    # ---- Needed for the example ---- #
    @property
    def name_dof(self):
        return ["cn", "f", "a", "tau1", "km"]

    @property
    def nb_state(self):
        return 5

    @property
    def name(self):
        return self._name

    def system_dynamics(
        self, cn: MX, f: MX, a: MX, tau1: MX, km: MX, t: MX, t_stim_prev: list[MX]
    ) -> MX:
        """
        The system dynamics is the function that describes the model.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        a: MX
            The value of the scaling factor (unitless)
        tau1: MX
            The value of the time_state_force_no_cross_bridge (ms)
        km: MX
            The value of the cross_bridges (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        # from Ding's 2003 article
        r0 = km + MX(self.r0_km_relationship)  # Simplification
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev)  # Equation n°1
        f_dot = self.f_dot_fun(cn, f, a, tau1, km)  # Equation n°2
        a_dot = self.a_dot_fun(a, f)  # Equation n°5
        tau1_dot = self.tau1_dot_fun(tau1, f)  # Equation n°9
        km_dot = self.km_dot_fun(km, f)  # Equation n°11

        return vertcat(cn_dot, f_dot, a_dot, tau1_dot, km_dot)

    def exp_time_fun(self, t: MX, t_stim_i: MX):
        """
        Parameters
        ----------
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_i: MX
            Time when the stimulation i occurred (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return exp(-(t - t_stim_i) / self.tauc)  # Eq from [1]

    def ri_fun(self, r0: MX, time_between_stim: MX):
        """
        Parameters
        ----------
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        time_between_stim: MX
            Time between the last stimulation i and the current stimulation i (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        return 1 + (r0 - 1) * exp(time_between_stim / self.tauc)  # Eq from [1]

    def cn_sum_fun(self, r0: MX, t: MX, t_stim_prev: list[MX]):
        """
        Parameters
        ----------
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)

        Returns
        -------
        A part of the n°1 equation
        """
        sum_multiplier = 0

        for i in range(len(t_stim_prev)):  # Eq from [1]
            if i == 0:  # Eq from Bakir et al.
                ri = 1
            else:
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)

            exp_time = self.exp_time_fun(t, t_stim_prev[i])
            sum_multiplier += ri * exp_time
        return sum_multiplier

    def cn_dot_fun(self, cn: MX, r0: MX, t: MX, t_stim_prev: list[MX]):
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        r0: MX
            Mathematical term characterizing the magnitude of enhancement in CN from the following stimuli (unitless)
        t: MX
            The current time at which the dynamics is evaluated (ms)
        t_stim_prev: list[MX]
            The list of the time of the previous stimulations (ms)

        Returns
        -------
        The value of the derivative ca_troponin_complex (unitless)
        """
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev)

        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)  # Eq(1)

    def f_dot_fun(self, cn: MX, f: MX, a: MX, tau1: MX, km: MX):
        """
        Parameters
        ----------
        cn: MX
            The previous step value of ca_troponin_complex (unitless)
        f: MX
            The previous step value of force (N)
        a: MX
            The previous step value of scaling factor (unitless)
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        km: MX
            The previous step value of cross_bridges (unitless)

        Returns
        -------
        The value of the derivative force (N)
        """
        return a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn))))  # Eq(2)

    def a_dot_fun(self, a: MX, f: MX):
        """
        Parameters
        ----------
        a: MX
            The previous step value of scaling factor (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative scaling factor (unitless)
        """
        return -(a - self.a_rest) / self.tau_fat + self.alpha_a * f  # Eq(5)

    def tau1_dot_fun(self, tau1: MX, f: MX):
        """
        Parameters
        ----------
        tau1: MX
            The previous step value of time_state_force_no_cross_bridge (ms)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative time_state_force_no_cross_bridge (ms)
        """
        return -(tau1 - self.tau1_rest) / self.tau_fat + self.alpha_tau1 * f  # Eq(9)

    def km_dot_fun(self, km: MX, f: MX):
        """
        Parameters
        ----------
        km: MX
            The previous step value of cross_bridges (unitless)
        f: MX
            The previous step value of force (N)

        Returns
        -------
        The value of the derivative cross_bridges (unitless)
        """
        return -(km - self.km_rest) / self.tau_fat + self.alpha_km * f  # Eq(11)


def custom_dynamics(
    states: MX,
    controls: MX,
    parameters: MX,
    nlp: NonLinearProgram,
    t=None,
) -> DynamicsEvaluation:
    """
    Functional electrical stimulation dynamic

    Parameters
    ----------
    states: MX | SX
        The state of the system CN, F, A, Tau1, Km
    controls: MX | SX
        The controls of the system, none
    parameters: MX | SX
        The parameters acting on the system, final time of each phase
    nlp: NonLinearProgram
        A reference to the phase
    t: MX
        Current node time, this t is used to set the dynamics and as to be a symbolic
    Returns
    -------
    The derivative of the states in the tuple[MX | SX]] format
    """

    t_stim_prev = []  # Every stimulation instant before the current phase, i.e.: the beginning of each phase

    for i in range(nlp.phase_idx+1):
        t_stim_prev.append(sum1(nlp.parameters.mx[0: i]))

    return DynamicsEvaluation(
        dxdt=nlp.model.system_dynamics(
            cn=states[0],
            f=states[1],
            a=states[2],
            tau1=states[3],
            km=states[4],
            t=t,
            t_stim_prev=t_stim_prev,
        ),
        defects=None,
    )


def custom_configure_dynamics_function(ocp, nlp, **extra_params):
    """
    Configure the dynamics of the system

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    **extra_params: t
        t: MX
            Current node time
    """

    global dynamics_eval_horzcat
    nlp.parameters = ocp.v.parameters_in_list
    DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

    # Gets the t0 time for the current phase
    t0_phase_in_ocp = sum1(nlp.parameters.mx[0: nlp.phase_idx])
    # Gets every time node for the current phase

    for i in range(nlp.ns):
        t_node_in_phase = nlp.parameters.mx[nlp.phase_idx] / (nlp.ns + 1) * i
        t_node_in_ocp = t0_phase_in_ocp + t_node_in_phase
        extra_params["t"] = t_node_in_ocp

        dynamics_eval = custom_dynamics(
            nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx, nlp, **extra_params
        )

        dynamics_eval_horzcat = horzcat(dynamics_eval.dxdt) if i == 0 else horzcat(dynamics_eval_horzcat, dynamics_eval.dxdt)

    nlp.dynamics_func = Function(
        "ForwardDyn",
        [nlp.states["scaled"].mx_reduced, nlp.controls["scaled"].mx_reduced, nlp.parameters.mx],
        [dynamics_eval_horzcat],
        ["x", "u", "p"],
        ["xdot"],
    )


def declare_ding_variables(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.
    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """
    configure_ca_troponin_complex(ocp, nlp, as_states=True, as_controls=False)
    configure_force(ocp, nlp, as_states=True, as_controls=False)
    configure_scaling_factor(ocp, nlp, as_states=True, as_controls=False)
    configure_time_state_force_no_cross_bridge(ocp, nlp, as_states=True, as_controls=False)
    configure_cross_bridges(ocp, nlp, as_states=True, as_controls=False)

    t = MX.sym("t")  # t needs a symbolic value to start computing in custom_configure_dynamics_function

    custom_configure_dynamics_function(ocp, nlp, t=t)


def configure_ca_troponin_complex(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable of the Ca+ troponin complex (unitless)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "Cn"
    name_cn = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_cn,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_force(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable of the force (N)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "F"
    name_f = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_f,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_scaling_factor(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable of the scaling factor (N/ms)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "A"
    name_a = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_a,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_time_state_force_no_cross_bridge(
    ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False
):
    """
    Configure a new variable for time constant of force decline at the absence of strongly bound cross-bridges (ms)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "Tau1"
    name_tau1 = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_tau1,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def configure_cross_bridges(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
    """
    Configure a new variable for sensitivity of strongly bound cross-bridges to Cn (unitless)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    as_states: bool
        If the generalized coordinates should be a state
    as_controls: bool
        If the generalized coordinates should be a control
    as_states_dot: bool
        If the generalized velocities should be a state_dot
    """
    name = "Km"
    name_km = [name]
    ConfigureProblem.configure_new_variable(
        name,
        name_km,
        ocp,
        nlp,
        as_states,
        as_controls,
        as_states_dot,
    )


def prepare_ocp(
        n_stim: int,
        time_min: list,
        time_max: list,
        stim_freq: int,
        ode_solver: OdeSolver = OdeSolver.RK2(n_integration_steps=1),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    ding_models = [DingModel() for i in range(n_stim)]  # Gives DingModel as model for n phases
    n_shooting = [5 for i in range(n_stim)]  # Gives m node shooting for my n phases problem
    final_time = [0.01 for i in range(n_stim)]  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(declare_ding_variables, dynamic_function=custom_dynamics, phase=i)

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(n_stim):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    # Frequency test
    # for i in range(n_stim):
    #     constraints.add(
    #         ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1, max_bound=0.01, phase=i
    #     )

    objective_functions = ObjectiveList()
    # Objective function to target force
    # for i in range(n_stim):
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, target=250, key="F", node=Node.END, quadratic=True, weight=1,
        phase=9)

    # Objective function to minimize muscle fatigue
    # for i in range(n_stim):
    #     objective_functions.add(
    #         ObjectiveFcn.Mayer.MINIMIZE_STATE, target=3009, key="A", node=Node.END, quadratic=True, weight=1,
    #         phase=i)

    # Sets the bound for all the phases
    x_bounds = BoundsList()

    x_min_start = ding_models[0].standard_rest_values()  # Model initial values
    x_max_start = ding_models[0].standard_rest_values()  # Model initial values

    # Model execution lower bound values (Cn, F, Tau1, Km, cannot be lower than their initial values)
    x_min_middle = ding_models[0].standard_rest_values()
    x_min_middle[2] = 0  # Model execution lower bound values (A, will decrease from fatigue and cannot be lower than 0)
    x_min_end = x_min_middle

    x_max_middle = ding_models[0].standard_rest_values()
    x_max_middle[0:2] = 1000
    x_max_middle[3:5] = 1
    x_max_end = x_max_middle

    x_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    x_min_start = x_min_middle
    x_max_start = x_max_middle

    x_after_start_min = np.concatenate((x_min_start, x_min_middle, x_min_end), axis=1)
    x_after_start_max = np.concatenate((x_max_start, x_max_middle, x_max_end), axis=1)

    for i in range(n_stim):
        if i == 0:
            x_bounds.add(
                x_start_min, x_start_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            )
        else:
            x_bounds.add(
                x_after_start_min,
                x_after_start_max,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

    x_init = InitialGuessList()
    for i in range(n_stim):
        x_init.add(ding_models[0].standard_rest_values())

    u_bounds = BoundsList()
    for i in range(n_stim):
        u_bounds.add([], [])

    u_init = InitialGuessList()
    for i in range(n_stim):
        u_init.add([])

    return OptimalControlProgram(
        ding_models,
        dynamics,
        n_shooting,
        final_time,  # frequency
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        control_type=ControlType.NONE,
        use_sx=True,
    )


def test_main_control_type_none():
    """
    Prepare and solve and animate a reaching task ocp
    """
    # number of stimulation corresponding to phases
    n = 20
    # minimum time between two phase (stimulation)
    time_min = [0.01 for _ in range(n)]
    # maximum time between two phase (stimulation)
    time_max = [0.1 for _ in range(n)]
    ocp = prepare_ocp(n_stim=n, time_min=time_min, time_max=time_max, stim_freq=33)

    # ocp = prepare_ocp(n_stim=n, stim_freq=33)

    ocp.print(to_console=True, to_graph=True)



    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # re-integrate the solution
    # simulate()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -143.5854887928483)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.37791617, 3.70167396, 10.0, 10.0]), decimal=2)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.37675299, -3.40771446, 10.0, 10.0]), decimal=2)
    # initial and final controls
    np.testing.assert_almost_equal(
        tau[:, 0], np.array([-4.52595667e-02, 9.25475333e-01, -4.34001849e-08, -9.24667407e01]), decimal=2
    )
    np.testing.assert_almost_equal(
        tau[:, -2], np.array([4.42976253e-02, 1.40077846e00, -7.28864793e-13, 9.24667396e01]), decimal=2
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)

