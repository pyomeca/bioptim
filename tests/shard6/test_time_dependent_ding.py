from typing import Callable
import numpy as np
import pytest
from casadi import MX, vertcat, exp

import matplotlib.pyplot as plt

from bioptim import (
    BiMapping,
    BoundsList,
    ConfigureProblem,
    ConstraintFcn,
    ConstraintList,
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
    PhaseDynamics,
    SolutionMerge,
)


class Model:
    def __init__(self):
        self._name = None
        self.a_rest = 3000
        self.tau1_rest = 0.05
        self.km_rest = 0.1
        self.tau2 = 0.06
        self.r0_km_relationship = 0.014
        self.tauc = 0.02

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
        return ["Cn", "F"]

    @property
    def nb_state(self) -> int:
        return 2

    @property
    def name(self) -> None | str:
        return self._name

    @staticmethod
    def standard_rest_values() -> np.array:
        return np.array([[0], [0]])

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
        return vertcat(cn_dot, f_dot)

    def exp_time_fun(self, t: MX, t_stim_i: MX) -> MX | float:
        return exp(-(t - t_stim_i) / self.tauc)

    def ri_fun(self, r0: MX | float, time_between_stim: MX) -> MX | float:
        return 1 + (r0 - 1) * exp(-time_between_stim / self.tauc)

    def cn_sum_fun(self, r0: MX | float, t: MX, t_stim_prev: list[MX]) -> MX | float:
        sum_multiplier = 0
        if len(t_stim_prev) == 1:
            ri = 1
            exp_time = self.exp_time_fun(t, t_stim_prev[0])
            sum_multiplier += ri * exp_time
        else:
            for i in range(1, len(t_stim_prev)):
                previous_phase_time = t_stim_prev[i] - t_stim_prev[i - 1]
                ri = self.ri_fun(r0, previous_phase_time)
                exp_time = self.exp_time_fun(t, t_stim_prev[i])
                sum_multiplier += ri * exp_time
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
        stochastic_variables: MX,
        nlp: NonLinearProgram,
        stim_apparition=None,
    ) -> DynamicsEvaluation:
        return DynamicsEvaluation(
            dxdt=nlp.model.system_dynamics(
                cn=states[0],
                f=states[1],
                t=time,
                t_stim_prev=stim_apparition,
            ),
            defects=None,
        )

    def declare_ding_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
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

        stim_apparition = [ocp.node_time(phase_idx=i, node_idx=0) for i in range(nlp.phase_idx + 1)]
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics, stim_apparition=stim_apparition, allow_free_variables=True)


def prepare_ocp(
    model: Model = None,
    n_stim: int = None,
    n_shooting: int = None,
    final_time: int | float = None,
    time_min: int | float = None,
    time_max: int | float = None,
    time_bimapping: bool = None,
    use_sx: bool = True,
):
    models = [model] * n_stim
    n_shooting = [n_shooting] * n_stim

    constraints = ConstraintList()
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

    phase_time_bimapping = None
    if time_bimapping is True:
        phase_time_bimapping = BiMapping(to_second=[0 for _ in range(n_stim)], to_first=[0])

    dynamics = DynamicsList()
    for i in range(n_stim):
        dynamics.add(
            models[i].declare_ding_variables,
            dynamic_function=dynamics,
            expand_dynamics=True,
            expand_continuity=False,
            phase=i,
            phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        )

    x_bounds = BoundsList()
    starting_bounds, min_bounds, max_bounds = (
        np.array([[0], [0]]),
        np.array([[0], [0]]),
        np.array([[0], [0]]),
    )

    variable_bound_list = ["Cn", "F"]

    max_bounds[0] = 400
    max_bounds[1] = 400

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
        target=200,
        phase=n_stim - 1,
    )

    return OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time_phase,
        time_phase_mapping=phase_time_bimapping,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        constraints=constraints,
        use_sx=use_sx,
        ode_solver=OdeSolver.RK4(n_integration_steps=1, allow_free_variables=True),
    )


@pytest.mark.parametrize("time_mapping", [False, True])
@pytest.mark.parametrize("use_sx", [False, True])
def test_time_dependent_ding(time_mapping, use_sx):
    ocp = prepare_ocp(
        model=Model(),
        n_stim=5,
        n_shooting=10,
        final_time=1,
        time_min=0.01,
        time_max=0.1,
        time_bimapping=time_mapping,
        use_sx=use_sx,
    )

    sol = ocp.solve()

    # if time_mapping:
    #     # Check cost
    #     f = np.array(sol.cost)
    #     np.testing.assert_equal(f.shape, (1, 1))
    #     np.testing.assert_almost_equal(f[0, 0], 4.44158452139091e-17)
    #
    #     # Check finishing time
    #     np.testing.assert_almost_equal(np.cumsum([t[-1] for t in sol.times])[-1], 0.17539096257753187)
    #
    #     # Check constraints
    #     g = np.array(sol.constraints)
    #     np.testing.assert_equal(g.shape, (113, 1))
    #
    #     # Check state values
    #     np.testing.assert_almost_equal(
    #         sol.decision_states(to_merge=SolutionMerge.NODES)[0]["F"],
    #         np.array(
    #             [
    #                 0,
    #                 4.09378412,
    #                 10.79399074,
    #                 18.01733422,
    #                 25.30841967,
    #                 32.48606806,
    #                 39.45727338,
    #                 46.16593939,
    #                 52.57386063,
    #                 58.65231534,
    #                 64.37791185,
    #             ]
    #         ),
    #         decimal=8,
    #     )
    #     np.testing.assert_almost_equal(
    #         sol.decision_states(to_merge=SolutionMerge.NODES)[4]["F"],
    #         np.array(
    #             [
    #                 180.39020796,
    #                 182.41999723,
    #                 184.710201,
    #                 187.06384558,
    #                 189.38056784,
    #                 191.59929805,
    #                 193.67765183,
    #                 195.58295877,
    #                 197.28783499,
    #                 198.76781029,
    #                 200.00000001,
    #             ]
    #         ),
    #         decimal=8,
    #     )

    # else:
        # # Check cost
        # f = np.array(sol.cost)
        # np.testing.assert_equal(f.shape, (1, 1))
        # np.testing.assert_almost_equal(f[0, 0], 3.433583564688405e-16)
        #
        # # Check finishing time
        # np.testing.assert_almost_equal(np.cumsum([t[-1] for t in sol.times])[-1], 0.1747389841117835)
        #
        # # Check constraints
        # g = np.array(sol.constraints)
        # np.testing.assert_equal(g.shape, (113, 1))
        #
        # # Check state values
        # np.testing.assert_almost_equal(
        #     sol.decision_states(to_merge=SolutionMerge.NODES)[0]["F"],
        #     np.array(
        #         [
        #             0.0,
        #             8.46374155,
        #             20.70026662,
        #             32.95186563,
        #             44.60499695,
        #             55.41568917,
        #             65.24356677,
        #             73.98606604,
        #             81.55727281,
        #             87.88197209,
        #             92.89677605,
        #         ]
        #     ),
        #     decimal=8,
        # )
        # np.testing.assert_almost_equal(
        #     sol.decision_states(to_merge=SolutionMerge.NODES)[4]["F"],
        #     np.array(
        #         [
        #             177.7799946,
        #             180.366933,
        #             183.0257059,
        #             185.65708359,
        #             188.20320576,
        #             190.62531536,
        #             192.89440721,
        #             194.98671017,
        #             196.88128256,
        #             198.5586481,
        #             199.99999998,
        #         ]
        #     ),
        #     decimal=8,
        # )

    force_vector = sol.decision_states(to_merge=SolutionMerge.NODES)
    for i in range(len(force_vector)):
        force_vector[i] = force_vector[i]["F"][0]
        if i != 0:
            force_vector[i] = force_vector[i][1:]
    force_vector = [item for row in force_vector for item in row]


    cn_vector = sol.decision_states(to_merge=SolutionMerge.NODES)
    for i in range(len(cn_vector)):
        cn_vector[i] = cn_vector[i]["Cn"][0]
        if i != 0:
            cn_vector[i] = cn_vector[i][1:]
    cn_vector = [item for row in cn_vector for item in row]

    time_vector = sol.times
    for i in range(len(time_vector)):
        time_vector[i] = time_vector[i][::2]
        if i != 0:
            time_vector[i] = time_vector[i][1:]
    time_vector = [item for row in time_vector for item in row]
    time_vector = np.cumsum(time_vector)

    plt.plot(time_vector, force_vector)
    plt.show()

    plt.plot(time_vector, cn_vector)
    plt.show()
