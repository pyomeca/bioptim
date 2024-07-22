import pytest

from typing import Callable
import numpy as np
import numpy.testing as npt
from casadi import DM, MX, SX, vertcat, exp

from bioptim import (
    BoundsList,
    ConfigureProblem,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsEvaluation,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    Node,
    NonLinearProgram,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    PenaltyController,
    PhaseDynamics,
    SolutionMerge,
    VariableScaling,
)


class Model:
    def __init__(self, time_as_states: bool = False):
        self._name = None
        self.a_rest = 3000
        self.tau1_rest = 0.05
        self.km_rest = 0.1
        self.tau2 = 0.06
        self.r0_km_relationship = 0.014
        self.tauc = 0.02
        self.time_as_states = time_as_states
        self.pulse_apparition_time = None

    def serialize(self) -> tuple[Callable, dict]:
        return (
            Model,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
            },
        )

    @property
    def name_dof(self) -> list[str]:
        return ["Cn", "F", "time"] if self.time_as_states else ["Cn", "F"]

    @property
    def nb_state(self) -> int:
        return 3 if self.time_as_states else 2

    @property
    def name(self) -> None | str:
        return self._name

    def standard_rest_values(self) -> np.array:
        return np.array([[0], [0], [0]]) if self.time_as_states else np.array([[0], [0]])

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        t_stim_prev: list[MX] | list[float] = None,
    ) -> MX:
        r0 = self.km_rest + self.r0_km_relationship
        cn_dot = self.cn_dot_fun(cn, r0, t, t_stim_prev=t_stim_prev)
        f_dot = self.f_dot_fun(cn, f, self.a_rest, self.tau1_rest, self.km_rest)
        return vertcat(cn_dot, f_dot, 1) if self.time_as_states else vertcat(cn_dot, f_dot)

    def exp_time_fun(self, t: MX, t_stim_i: MX) -> MX | float:
        return exp(-(t - t_stim_i) / self.tauc)

    def ri_fun(self, r0: MX | float, time_between_stim: MX) -> MX | float:
        return 1 + (r0 - 1) * exp(-time_between_stim / self.tauc)

    def cn_sum_fun(self, r0: MX | float, t: MX, t_stim_prev: list[MX]) -> MX | float:
        sum_multiplier = 0
        if len(t_stim_prev) == 1:
            ri = 1
            exp_time = self.exp_time_fun(t, t_stim_prev[0])  # Part of Eq n°1
            sum_multiplier += ri * exp_time  # Part of Eq n°1
        else:
            for i in range(1, len(t_stim_prev)):
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)  # Part of Eq n°1
                exp_time = self.exp_time_fun(t, t_stim_prev[i])  # Part of Eq n°1
                sum_multiplier += ri * exp_time  # Part of Eq n°1
        return sum_multiplier

    def cn_dot_fun(self, cn: MX, r0: MX | float, t: MX, t_stim_prev: list[MX]) -> MX | float:
        sum_multiplier = self.cn_sum_fun(r0, t, t_stim_prev=t_stim_prev)
        return (1 / self.tauc) * sum_multiplier - (cn / self.tauc)

    def f_dot_fun(self, cn: MX, f: MX, a: MX | float, tau1: MX | float, km: MX | float) -> MX | float:
        return a * (cn / (km + cn)) - (f / (tau1 + self.tau2 * (cn / (km + cn))))

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:

        stim_prev = get_stim_prev(nlp, parameters, nlp.phase_idx)

        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(
                cn=states[0],
                f=states[1],
                t=states[2] if nlp.model.time_as_states else time,
                t_stim_prev=stim_prev,
            ),
            defects=None,
        )

    def declare_ding_variables(
        self, ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries: dict[str, np.ndarray] = None
    ):
        name = "Cn"
        name_cn = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_cn,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
        )

        name = "F"
        name_f = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_f,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
        )

        if self.time_as_states:
            name = "time"
            name_time = [name]
            ConfigureProblem.configure_new_variable(
                name,
                name_time,
                ocp,
                nlp,
                as_states=True,
                as_controls=False,
            )

        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics)

    def set_pulse_apparition_time(self, value: list[MX], kwargs: dict = None):
        """
        Sets the pulse apparition time for each pulse (phases) according to the ocp parameter "pulse_apparition_time"

        Parameters
        ----------
        value: list[MX]
            The pulse apparition time list (s)
        kwargs: dict
            The kwargs of the ocp
        """
        self.pulse_apparition_time = value


def get_stim_prev(nlp: NonLinearProgram, parameters: MX, idx: int) -> list[float]:
    """
    Get the nlp list of previous stimulation apparition time

    Parameters
    ----------
    nlp: NonLinearProgram
        The NonLinearProgram of the ocp of the current phase
    parameters: MX
        The parameters of the ocp
    idx: int
        The index of the current phase

    Returns
    -------
    The list of previous stimulation time
    """
    t_stim_prev = []
    for j in range(parameters.shape[0]):
        if "pulse_apparition_time" in nlp.parameters.cx[j].str():
            t_stim_prev.append(parameters[j])
        if len(t_stim_prev) > idx:
            break
    return t_stim_prev


class CustomConstraint:
    @staticmethod
    def pulse_time_apparition_as_phase(controller: PenaltyController) -> MX | SX:
        return controller.time.cx - controller.parameters["pulse_apparition_time"].cx[controller.phase_idx]

    @staticmethod
    def equal_to_first_pulse_interval_time(controller: PenaltyController) -> MX | SX:
        if controller.ocp.n_phases <= 1:
            RuntimeError("There is only one phase, the bimapping constraint is not possible")

        first_phase_tf = controller.ocp.node_time(0, controller.ocp.nlp[controller.phase_idx].ns)
        current_phase_tf = controller.ocp.nlp[controller.phase_idx].node_time(
            controller.ocp.nlp[controller.phase_idx].ns
        )
        return first_phase_tf - current_phase_tf


def prepare_ocp(
    model: Model = None,
    n_stim: int = None,
    n_shooting: int = None,
    final_time: int | float = None,
    pulse_event: dict = None,
    use_sx: bool = True,
):
    models = [model] * n_stim
    n_shooting = [n_shooting] * n_stim
    time_min = pulse_event["time_min"]
    time_max = pulse_event["time_max"]
    time_bimapping = pulse_event["time_bimapping"]

    constraints = ConstraintList()
    if time_min and time_max:
        for i in range(n_stim):
            constraints.add(
                ConstraintFcn.TIME_CONSTRAINT,
                node=Node.END,
                min_bound=time_min,
                max_bound=time_max,
                phase=i,
            )

    step_phase = final_time / n_stim
    final_time_phase = [step_phase] * n_stim

    parameters = ParameterList(use_sx=use_sx)
    parameters_bounds = BoundsList()
    parameters_init = InitialGuessList()
    constraints = ConstraintList()
    if time_min:
        parameters.add(
            name="pulse_apparition_time",
            function=model.set_pulse_apparition_time,
            size=n_stim,
            scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
        )

        if time_min and time_max:
            time_min_list = [time_min * n for n in range(n_stim)]
            time_max_list = [time_max * n for n in range(n_stim)]
        else:
            time_min_list = [0] * n_stim
            time_max_list = [100] * n_stim
        parameters_bounds.add(
            "pulse_apparition_time",
            min_bound=np.array(time_min_list),
            max_bound=np.array(time_max_list),
            interpolation=InterpolationType.CONSTANT,
        )

        parameters_init["pulse_apparition_time"] = np.array([0] * n_stim)

        for i in range(n_stim):
            constraints.add(CustomConstraint.pulse_time_apparition_as_phase, node=Node.START, phase=i, target=0)

    if time_bimapping and time_min and time_max:
        for i in range(n_stim):
            constraints.add(CustomConstraint.equal_to_first_pulse_interval_time, node=Node.START, target=0, phase=i)

    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(
            models[i].declare_ding_variables,
            dynamic_function=models[i].dynamics,
            expand_dynamics=True,
            expand_continuity=False,
            phase=i,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        )

    x_bounds = BoundsList()
    starting_bounds, min_bounds, max_bounds = (
        np.array([[0], [0], [0]]),
        np.array([[0], [0], [0]]),
        np.array([[0], [0], [0]]),
    )

    variable_bound_list = ["Cn", "F", "time"] if model.time_as_states else ["Cn", "F"]

    max_bounds[0] = 400
    max_bounds[1] = 400
    if model.time_as_states:
        max_bounds[2] = 1

    starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
    starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
    middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
    middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

    for i in range(n_stim):
        for j in range(len(variable_bound_list)):
            if i == 0:
                x_bounds.add(
                    variable_bound_list[j],
                    min_bound=np.array([starting_bounds_min[j]]),
                    max_bound=np.array([starting_bounds_max[j]]),
                    phase=i,
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )
            else:
                x_bounds.add(
                    variable_bound_list[j],
                    min_bound=np.array([middle_bound_min[j]]),
                    max_bound=np.array([middle_bound_max[j]]),
                    phase=i,
                    interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
                )

    x_init = InitialGuessList()
    for i in range(n_stim):
        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], np.array([0]), phase=i)

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        node=Node.END,
        key="F",
        quadratic=True,
        weight=1,
        target=180,
        phase=n_stim - 1,
    )
    for i in range(n_stim):
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME,
            weight=0.001 / n_shooting[i],
            min_bound=time_min,
            max_bound=time_max,
            quadratic=True,
            phase=i,
        )

    return OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time_phase,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        constraints=constraints,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        ode_solver=OdeSolver.RK4(n_integration_steps=1),
    )


problem_dict = {
    "0": {"time_as_states": False, "use_sx": False, "time_mapping": False},
    "1": {"time_as_states": False, "use_sx": False, "time_mapping": True},
    "2": {"time_as_states": False, "use_sx": True, "time_mapping": False},
    "3": {"time_as_states": False, "use_sx": True, "time_mapping": True},
    "4": {"time_as_states": True, "use_sx": False, "time_mapping": False},
    "5": {"time_as_states": True, "use_sx": False, "time_mapping": True},
    "6": {"time_as_states": True, "use_sx": True, "time_mapping": False},
    "7": {"time_as_states": True, "use_sx": True, "time_mapping": True},
}

result_dict = {
    "0": {
        "cost_value": 3.6450263096541647e-06,
        "final_time": DM(0.142389),
        "constraint_shape": (113, 1),
        "force_values": np.array(
            [
                0.0,
                4.91287302,
                12.71221056,
                20.9783046,
                29.21889376,
                37.24069434,
                44.94387131,
                52.26680964,
                59.16593448,
                65.60694336,
                71.56061724,
                71.56061724,
                73.86614209,
                76.25911509,
                78.69900023,
                81.159642,
                83.62308848,
                86.07640907,
                88.50991551,
                90.91610229,
                93.28898275,
                95.6236555,
                95.6236555,
                97.66937716,
                99.73432545,
                101.80557923,
                103.87374525,
                105.93174765,
                107.97409359,
                109.99640605,
                111.99511487,
                113.96724554,
                115.9102707,
                115.9102707,
                117.67253901,
                119.43720944,
                121.19815912,
                122.95064403,
                124.69089614,
                126.41585626,
                128.12299105,
                129.81016411,
                131.47554311,
                133.11753126,
                133.11753126,
                140.91607082,
                148.36108307,
                155.25810302,
                161.49281132,
                166.97477641,
                171.61834036,
                175.33536648,
                178.03356582,
                179.61866967,
                179.99999965,
            ]
        ),
        "parameters": np.array([0.0, 0.03986182, 0.0555412, 0.06929909, 0.08184975]),
    },
    "1": {
        "cost_value": 3.097093445613913e-05,
        "final_time": DM(0.375203),
        "constraint_shape": (118, 1),
        "force_values": np.array(
            [
                8.33996926e-18,
                1.13421623e01,
                2.69060582e01,
                4.19059492e01,
                5.56416292e01,
                6.78112639e01,
                7.82147745e01,
                8.66897358e01,
                9.31019673e01,
                9.73578259e01,
                9.94265644e01,
                9.94265644e01,
                1.05824231e02,
                1.14879046e02,
                1.23656183e02,
                1.31502167e02,
                1.38092064e02,
                1.43192896e02,
                1.46609195e02,
                1.48177070e02,
                1.47779196e02,
                1.45369829e02,
                1.45369829e02,
                1.48158367e02,
                1.54156699e02,
                1.60131881e02,
                1.65373860e02,
                1.69528402e02,
                1.72341104e02,
                1.73598122e02,
                1.73118660e02,
                1.70769642e02,
                1.66491303e02,
                1.66491303e02,
                1.67651755e02,
                1.72218608e02,
                1.76874128e02,
                1.80884375e02,
                1.83881425e02,
                1.85600718e02,
                1.85820067e02,
                1.84351681e02,
                1.81056888e02,
                1.75871989e02,
                1.75871989e02,
                1.76283259e02,
                1.80209227e02,
                1.84276380e02,
                1.87738275e02,
                1.90220346e02,
                1.91453296e02,
                1.91211148e02,
                1.89302997e02,
                1.85587734e02,
                1.80000000e02,
            ]
        ),
        "parameters": np.array([-6.59076798e-12, 7.50405155e-02, 1.50081031e-01, 2.25121547e-01, 3.00162062e-01]),
    },
    "2": {
        "cost_value": 3.6450269758663618e-06,
        "final_time": DM(0.142389),
        "constraint_shape": (113, 1),
        "force_values": np.array(
            [
                0.0,
                4.9128671,
                12.71219683,
                20.97828354,
                29.2188661,
                37.2406609,
                44.94383294,
                52.26676725,
                59.16588903,
                65.60689584,
                71.56056869,
                71.56056869,
                73.86609751,
                76.25907436,
                78.69896321,
                81.15960856,
                83.62305847,
                86.07638235,
                88.50989194,
                90.91608172,
                93.28896502,
                95.62364047,
                95.62364047,
                97.66936362,
                99.7343134,
                101.80556867,
                103.87373614,
                105.93173996,
                107.97408727,
                109.99640107,
                111.99511118,
                113.96724308,
                115.91026943,
                115.91026943,
                117.67253874,
                119.43721018,
                121.19816086,
                122.95064675,
                124.69089983,
                126.4158609,
                128.12299659,
                129.81017054,
                131.47555039,
                133.11753935,
                133.11753935,
                140.91608416,
                148.361101,
                155.25812453,
                161.49283518,
                166.97480117,
                171.61836437,
                175.33538787,
                178.03358247,
                179.61867925,
                179.99999965,
            ]
        ),
        "parameters": np.array([0.0, 0.03986179, 0.05554119, 0.06929908, 0.08184975]),
    },
    "3": {
        "cost_value": 3.097093445613913e-05,
        "final_time": DM(0.375203),
        "constraint_shape": (118, 1),
        "force_values": np.array(
            [
                8.33996945e-18,
                1.13421623e01,
                2.69060582e01,
                4.19059492e01,
                5.56416292e01,
                6.78112639e01,
                7.82147745e01,
                8.66897358e01,
                9.31019673e01,
                9.73578259e01,
                9.94265644e01,
                9.94265644e01,
                1.05824231e02,
                1.14879046e02,
                1.23656183e02,
                1.31502167e02,
                1.38092064e02,
                1.43192896e02,
                1.46609195e02,
                1.48177070e02,
                1.47779196e02,
                1.45369829e02,
                1.45369829e02,
                1.48158367e02,
                1.54156699e02,
                1.60131881e02,
                1.65373860e02,
                1.69528402e02,
                1.72341104e02,
                1.73598122e02,
                1.73118660e02,
                1.70769642e02,
                1.66491303e02,
                1.66491303e02,
                1.67651755e02,
                1.72218608e02,
                1.76874128e02,
                1.80884375e02,
                1.83881425e02,
                1.85600718e02,
                1.85820067e02,
                1.84351681e02,
                1.81056888e02,
                1.75871989e02,
                1.75871989e02,
                1.76283259e02,
                1.80209227e02,
                1.84276380e02,
                1.87738275e02,
                1.90220346e02,
                1.91453296e02,
                1.91211148e02,
                1.89302997e02,
                1.85587734e02,
                1.80000000e02,
            ]
        ),
        "parameters": np.array([-6.59076798e-12, 7.50405155e-02, 1.50081031e-01, 2.25121547e-01, 3.00162062e-01]),
    },
    "4": {
        "cost_value": 1.0238209212546622e-05,
        "final_time": DM(0.226438),
        "constraint_shape": (167, 1),
        "force_values": np.array(
            [
                0.0,
                11.18110153,
                26.56376841,
                41.4197328,
                55.05383108,
                67.16817584,
                77.5668464,
                86.0912597,
                92.60949859,
                97.0268919,
                99.30666483,
                99.30666483,
                101.09802666,
                104.10715624,
                107.53610736,
                111.11332262,
                114.71051176,
                118.25513811,
                121.70084701,
                125.01509607,
                128.17321306,
                131.15525353,
                131.15525353,
                133.09743685,
                135.14464151,
                137.24351881,
                139.36109254,
                141.47533855,
                143.57063758,
                145.63535374,
                147.6604454,
                149.63862037,
                151.56379741,
                151.56379741,
                153.36631845,
                155.19733386,
                157.03585171,
                158.86723678,
                160.68076135,
                162.46822259,
                164.22311231,
                165.94009309,
                167.6146537,
                169.24287475,
                169.24287475,
                177.01150936,
                184.23516793,
                190.39086484,
                195.16556184,
                198.27897921,
                199.44612493,
                198.38522854,
                194.85556103,
                188.71756489,
                180.00000005,
            ]
        ),
        "parameters": np.array([0.0, 0.07417641, 0.10371193, 0.12276609, 0.140635]),
    },
    "5": {
        "cost_value": 3.0970934444337856e-05,
        "final_time": DM(0.375203),
        "constraint_shape": (172, 1),
        "force_values": np.array(
            [
                8.60366279e-18,
                1.13421623e01,
                2.69060582e01,
                4.19059492e01,
                5.56416292e01,
                6.78112639e01,
                7.82147745e01,
                8.66897358e01,
                9.31019672e01,
                9.73578258e01,
                9.94265643e01,
                9.94265643e01,
                1.05824231e02,
                1.14879046e02,
                1.23656182e02,
                1.31502167e02,
                1.38092064e02,
                1.43192896e02,
                1.46609195e02,
                1.48177070e02,
                1.47779196e02,
                1.45369829e02,
                1.45369829e02,
                1.48158367e02,
                1.54156699e02,
                1.60131880e02,
                1.65373860e02,
                1.69528402e02,
                1.72341104e02,
                1.73598122e02,
                1.73118660e02,
                1.70769642e02,
                1.66491303e02,
                1.66491303e02,
                1.67651755e02,
                1.72218608e02,
                1.76874128e02,
                1.80884375e02,
                1.83881425e02,
                1.85600718e02,
                1.85820067e02,
                1.84351681e02,
                1.81056888e02,
                1.75871989e02,
                1.75871989e02,
                1.76283259e02,
                1.80209227e02,
                1.84276380e02,
                1.87738275e02,
                1.90220346e02,
                1.91453296e02,
                1.91211148e02,
                1.89302997e02,
                1.85587734e02,
                1.80000000e02,
            ]
        ),
        "parameters": np.array([-5.44721767e-11, 7.50405155e-02, 1.50081031e-01, 2.25121547e-01, 3.00162062e-01]),
    },
    "6": {
        "cost_value": 1.4744215248346423e-05,
        "final_time": DM(0.263452),
        "constraint_shape": (167, 1),
        "force_values": np.array(
            [
                0.0,
                13.57424936,
                31.59763422,
                48.47796448,
                63.43908151,
                76.11323369,
                86.23309996,
                93.58441051,
                98.02605765,
                99.53612147,
                98.25781905,
                98.25781905,
                100.27972949,
                104.59072551,
                109.42464742,
                114.33274892,
                119.12285032,
                123.6892281,
                127.96300922,
                131.89321109,
                135.43828112,
                138.56200793,
                138.56200793,
                140.62351239,
                142.97305789,
                145.44626478,
                147.9588152,
                150.46078541,
                152.9194609,
                155.31163833,
                157.61974082,
                159.82967973,
                161.92959782,
                161.92959782,
                163.82276943,
                165.79884198,
                167.80462596,
                169.80692028,
                171.78320461,
                173.7171525,
                175.59624289,
                177.41038965,
                179.15110768,
                180.8109814,
                180.8109814,
                187.38371743,
                193.62434218,
                198.8472007,
                202.67445215,
                204.78214558,
                204.85107231,
                202.57715671,
                197.71774649,
                190.16146754,
                180.00000007,
            ]
        ),
        "parameters": np.array([0.0, 0.08704349, 0.12749953, 0.15288344, 0.17558289]),
    },
    "7": {
        "cost_value": 3.097093444433773e-05,
        "final_time": DM(0.375203),
        "constraint_shape": (172, 1),
        "force_values": np.array(
            [
                8.60366226e-18,
                1.13421623e01,
                2.69060582e01,
                4.19059492e01,
                5.56416292e01,
                6.78112639e01,
                7.82147745e01,
                8.66897358e01,
                9.31019672e01,
                9.73578258e01,
                9.94265643e01,
                9.94265643e01,
                1.05824231e02,
                1.14879046e02,
                1.23656182e02,
                1.31502167e02,
                1.38092064e02,
                1.43192896e02,
                1.46609195e02,
                1.48177070e02,
                1.47779196e02,
                1.45369829e02,
                1.45369829e02,
                1.48158367e02,
                1.54156699e02,
                1.60131880e02,
                1.65373860e02,
                1.69528402e02,
                1.72341104e02,
                1.73598122e02,
                1.73118660e02,
                1.70769642e02,
                1.66491303e02,
                1.66491303e02,
                1.67651755e02,
                1.72218608e02,
                1.76874128e02,
                1.80884375e02,
                1.83881425e02,
                1.85600718e02,
                1.85820067e02,
                1.84351681e02,
                1.81056888e02,
                1.75871989e02,
                1.75871989e02,
                1.76283259e02,
                1.80209227e02,
                1.84276380e02,
                1.87738275e02,
                1.90220346e02,
                1.91453296e02,
                1.91211148e02,
                1.89302997e02,
                1.85587734e02,
                1.80000000e02,
            ]
        ),
        "parameters": np.array([-5.44721764e-11, 7.50405155e-02, 1.50081031e-01, 2.25121547e-01, 3.00162062e-01]),
    },
}


@pytest.mark.parametrize("test_index", [0, 1, 2, 3, 4, 5, 6, 7])
def test_time_dependent_ding(test_index):
    time_as_states = problem_dict[str(test_index)]["time_as_states"]
    use_sx = problem_dict[str(test_index)]["use_sx"]
    time_mapping = problem_dict[str(test_index)]["time_mapping"]

    ocp = prepare_ocp(
        model=Model(time_as_states=time_as_states),
        n_stim=5,
        n_shooting=10,
        final_time=1,
        pulse_event={"time_min": 0.01, "time_max": 0.1, "time_bimapping": time_mapping},
        use_sx=use_sx,
    )

    sol = ocp.solve()

    # Check cost
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], result_dict[str(test_index)]["cost_value"])

    # Check finishing time
    npt.assert_almost_equal(sol.decision_time()[-1][-1], result_dict[str(test_index)]["final_time"], decimal=6)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, result_dict[str(test_index)]["constraint_shape"])

    # Check state values
    npt.assert_almost_equal(
        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["F"][0],
        result_dict[str(test_index)]["force_values"],
        decimal=6,
    )

    # Check parameters
    npt.assert_almost_equal(
        sol.parameters["pulse_apparition_time"], result_dict[str(test_index)]["parameters"], decimal=8
    )
