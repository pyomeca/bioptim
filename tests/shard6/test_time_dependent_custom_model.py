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
                4.91287187,
                12.71220789,
                20.9783005,
                29.21888837,
                37.24068783,
                44.94386384,
                52.26680138,
                59.16592563,
                65.60693411,
                71.56060778,
                71.56060778,
                73.86613339,
                76.25910713,
                78.69899298,
                81.15963543,
                83.62308256,
                86.07640378,
                88.50991082,
                90.91609817,
                93.28897917,
                95.62365243,
                95.62365243,
                97.66937445,
                99.73432308,
                101.80557722,
                103.87374358,
                105.93174632,
                107.97409258,
                109.99640535,
                111.99511448,
                113.96724544,
                115.91027088,
                115.91027088,
                117.67253954,
                119.43721033,
                121.19816036,
                122.95064561,
                124.69089806,
                126.41585852,
                128.12299362,
                129.81016699,
                131.47554629,
                133.11753472,
                133.11753472,
                140.91607774,
                148.361093,
                155.25811533,
                161.49282523,
                166.97479102,
                171.61835466,
                175.3353793,
                178.03357585,
                179.61867546,
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
                4.91286843,
                12.71219991,
                20.97828826,
                29.2188723,
                37.24066839,
                44.94384154,
                52.26677675,
                59.16589922,
                65.60690649,
                71.56057957,
                71.56057957,
                73.86610789,
                76.25908428,
                78.69897272,
                81.15961767,
                83.62306721,
                86.07639074,
                88.5099,
                90.91608946,
                93.28897247,
                95.62364763,
                95.62364763,
                97.66937018,
                99.73431934,
                101.80557401,
                103.87374089,
                105.93174413,
                107.97409088,
                109.99640413,
                111.99511372,
                113.96724511,
                115.91027097,
                115.91027097,
                117.67254003,
                119.43721122,
                121.19816165,
                122.9506473,
                124.69090013,
                126.41586096,
                128.12299643,
                129.81017015,
                131.47554979,
                133.11753855,
                133.11753855,
                140.91608409,
                148.36110157,
                155.25812562,
                161.49283665,
                166.97480288,
                171.61836616,
                175.33538955,
                178.03358383,
                179.61868005,
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
        "final_time": DM(0.225507),
        "constraint_shape": (167, 1),
        "force_values": np.array(
            [
                0.0,
                11.21204467,
                26.62957125,
                41.51327447,
                55.16702526,
                67.29218558,
                77.69204358,
                86.20728054,
                92.70553426,
                97.09230071,
                99.33190884,
                99.33190884,
                101.06467541,
                103.99630391,
                107.34787577,
                110.85229555,
                114.3830314,
                117.86850371,
                121.26302392,
                124.53460348,
                127.65908519,
                130.61703311,
                130.61703311,
                132.54676835,
                134.57619321,
                136.65499999,
                138.75183789,
                140.84564979,
                142.92144352,
                144.96802091,
                146.97666686,
                148.94034785,
                150.85319898,
                150.85319898,
                152.66625467,
                154.50640365,
                156.35319394,
                158.19233006,
                160.01331269,
                161.80810148,
                163.57030968,
                165.29469477,
                166.97682283,
                168.61283982,
                168.61283982,
                176.42949832,
                183.69582846,
                189.89474865,
                194.71615473,
                197.88217871,
                199.1100246,
                198.11961433,
                194.67077098,
                188.62257727,
                180.00000005,
            ]
        ),
        "parameters": np.array([0.0, 0.07434242, 0.10332864, 0.12207148, 0.13986689]),
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
        "cost_value": 9.365334153318686e-06,
        "final_time": DM(0.217399),
        "constraint_shape": (167, 1),
        "force_values": np.array(
            [
                0.0,
                10.68601103,
                25.50816614,
                39.91461088,
                53.22543701,
                65.15428989,
                75.51740634,
                84.16777199,
                90.98113314,
                95.86190594,
                98.75890955,
                98.75890955,
                100.5808114,
                103.45551306,
                106.71576051,
                110.12125426,
                113.55560859,
                116.95219736,
                120.26824015,
                123.47373311,
                126.54605202,
                129.46706323,
                129.46706323,
                131.37049488,
                133.36277653,
                135.3996367,
                137.45300101,
                139.50378084,
                141.5382672,
                143.54615937,
                145.51941064,
                147.45151496,
                149.33704597,
                149.33704597,
                150.8815262,
                152.4458294,
                154.01732233,
                155.58674215,
                157.14704455,
                158.6927018,
                160.21925471,
                161.72301517,
                163.20086231,
                164.65009882,
                164.65009882,
                172.8399898,
                180.42862796,
                186.94042326,
                192.08416179,
                195.59577069,
                197.20379791,
                196.63596041,
                193.65340369,
                188.10661097,
                180.00000005,
            ]
        ),
        "parameters": np.array([0.0, 0.0715201, 0.09949487, 0.11759477, 0.13247013]),
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
