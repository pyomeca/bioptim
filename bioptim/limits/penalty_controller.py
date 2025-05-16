from typing import Any, Callable

from casadi import MX, SX, DM, vertcat

from ..dynamics.ode_solvers import OdeSolver
from ..misc.enums import ControlType, PhaseDynamics
from ..misc.mapping import BiMapping
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import OptimizationVariableList, OptimizationVariable


from ..misc.parameters_types import (
    Int,
    IntOptional,
    AnyList,
    CX,
)


class PenaltyController:
    """
    A placeholder for the required elements to compute a penalty (all time)
    For most of the part, PenaltyController will behave like NLP with the major difference that it will always
    return the states and controls form the current node_index (instead of all of them). If for some reason,
    one must access specific node that is not the current node_index, they can directly access the _nlp.
    Please note that this will likely result in free variables, which can be a pain to deal with...
    """

    def __init__(
        self,
        ocp,
        nlp: NonLinearProgram,
        t: AnyList,
        x: AnyList,
        u: AnyList,
        x_scaled: AnyList,
        u_scaled: AnyList,
        p: CX | AnyList,
        a: AnyList,
        a_scaled: AnyList,
        d: AnyList,
        node_index: IntOptional = None,
    ):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        t: list
            Time indices, maximum value being the number of shooting point + 1
        x: list
            References to the state variables
        u: list
            References to the control variables
        x_scaled: list
            References to the scaled state variables
        u_scaled: list
            References to the scaled control variables
        p: MX | SX | list
            References to the parameter variables
        a: list
            References to the algebraic_states variables
        a_scaled: list
            References to the scaled algebraic_states variables
        d: list
            References to the numerical timeseries
        node_index: int
            Current node index if nlp.phase_dynamics is SHARED_DURING_THE_PHASE,
            then node_index is expected to be set to 0
        """

        self._ocp: Any = ocp
        self._nlp: NonLinearProgram = nlp
        self.t = t
        self.x = x
        self.u = u
        self.x_scaled = x_scaled
        self.u_scaled = u_scaled
        self.a = a
        self.a_scaled = a_scaled
        self.p = vertcat(p) if p is not None else p
        self.d = d
        self.node_index = node_index
        self.cx_index_to_get = 0

    def __len__(self):
        return len(self.t)

    @property
    def ocp(self):
        return self._ocp

    @property
    def get_nlp(self):
        """
        This method returns the underlying nlp. Please note that acting directly with the nlp is not want you should do.
        Unless you see no way to access what you need otherwise, we strongly suggest that you use the normal path
        """
        return self._nlp

    @property
    def cx(self) -> CX | Callable:
        return self._nlp.cx

    @property
    def control_type(self) -> ControlType:
        return self._nlp.dynamics_type.control_type

    @property
    def ode_solver(self) -> OdeSolver:
        return self._nlp.dynamics_type.ode_solver

    @property
    def phase_idx(self) -> Int:
        return self._nlp.phase_idx

    @property
    def ns(self) -> Int:
        return self._nlp.ns

    @property
    def model(self):
        return self._nlp.model

    @property
    def t_span(self) -> OptimizationVariable:
        """
        Return the time span associated with the current node index. This value is the one that is used to integrate

        Returns
        -------

        """
        cx = vertcat(self.time.cx, self.dt.cx)

        tp = OptimizationVariableList(self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE)
        n_val = cx.shape[0]
        tp.append("t_span", cx=[cx, cx, cx], bimapping=BiMapping(to_second=range(n_val), to_first=range(n_val)))
        return tp["t_span"]

    @property
    def phases_dt(self) -> OptimizationVariable:
        """
        Return the delta time associated with all the phases
        """

        tp = OptimizationVariableList(self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE)
        n_val = self.ocp.dt_parameter.cx.shape[0]
        tp.append(
            "phases_dt",
            cx=[self.ocp.dt_parameter.cx, self.ocp.dt_parameter.cx, self.ocp.dt_parameter.cx],
            bimapping=BiMapping(to_second=range(n_val), to_first=range(n_val)),
        )

        return tp["phases_dt"]

    @property
    def dt(self) -> OptimizationVariable:
        """
        Return the delta time associated with the current phase
        """

        tp = OptimizationVariableList(self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE)
        n_val = self._nlp.dt.shape[0]
        tp.append(
            "dt",
            cx=[self._nlp.dt, self._nlp.dt, self._nlp.dt],
            bimapping=BiMapping(to_second=range(n_val), to_first=range(n_val)),
        )
        return tp["dt"]

    @property
    def time(self) -> OptimizationVariable:
        """
        Return the t0 associated with the current node index
        """

        tp = OptimizationVariableList(self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE)
        n_val = self._nlp.time_cx.shape[0]
        tp.append(
            "time",
            cx=[self._nlp.time_cx, self._nlp.time_cx, self._nlp.time_cx],
            bimapping=BiMapping(to_second=range(n_val), to_first=range(n_val)),
        )
        return tp["time"]

    @property
    def tf(self) -> OptimizationVariable:
        """
        Return the final time of the current phase
        """

        tp = OptimizationVariableList(self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE)
        n_val = self._nlp.tf.shape[0]
        tp.append(
            "tf",
            cx=[self._nlp.tf, self._nlp.tf, self._nlp.tf],
            bimapping=BiMapping(to_second=range(n_val), to_first=range(n_val)),
        )

        return tp["tf"]

    @property
    def states(self) -> OptimizationVariableList:
        """
        Return the states associated with the current node index

        Returns
        -------
        The states at node node_index
        """
        self._nlp.states.node_index = self.node_index
        out = self._nlp.states.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def controls(self) -> OptimizationVariableList:
        """
        Return the controls associated with the current node index

        Returns
        -------
        The controls at node node_index
        """
        self._nlp.controls.node_index = self.node_index
        out = self._nlp.controls.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def states_dot(self) -> OptimizationVariableList:
        """
        Return the states_dot associated with the current node index

        Returns
        -------
        The states_dot at node node_index
        """
        self._nlp.states_dot.node_index = self.node_index
        out = self._nlp.states_dot.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def algebraic_states(self) -> OptimizationVariableList:
        """
        Return the algebraic_states associated with the current node index
        Returns
        -------
        The algebraic_states at node node_index
        """
        self._nlp.algebraic_states.node_index = self.node_index
        out = self._nlp.algebraic_states.unscaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def integrated_values(self) -> OptimizationVariableList:
        """
        Return the values associated with the current node index
        Returns
        -------
        The integrated_values at node node_index
        """
        self._nlp.integrated_values.node_index = self.node_index
        out = self._nlp.integrated_values
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def numerical_timeseries(self) -> OptimizationVariableList:
        """
        Return the numerical_timeseries at node node_index=0.
        """
        self._nlp.numerical_timeseries.node_index = self.node_index
        out = self._nlp.numerical_timeseries
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def integrate(self) -> Callable:
        return self._nlp.dynamics[self.node_index]

    def integrate_extra_dynamics(self, dynamics_index) -> Callable:
        return self._nlp.extra_dynamics[dynamics_index][self.node_index]

    @property
    def dynamics(self) -> Callable:
        return self._nlp.dynamics_func

    def extra_dynamics(self, dynamics_index) -> Callable:
        return self._nlp.extra_dynamics_func[dynamics_index]

    @property
    def states_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled states associated with the current node index.

        Warning: Most of the time, the user does not want that states but the normal `states`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled states at node node_index
        """
        self._nlp.states.node_index = self.node_index
        out = self._nlp.states.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def controls_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled controls associated with the current node index.

        Warning: Most of the time, the user does not want that controls but the normal `controls`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled controls at node node_index
        """
        self._nlp.controls.node_index = self.node_index
        out = self._nlp.controls.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def states_dot_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled states_dot associated with the current node index.

        Warning: Most of the time, the user does not want that states but the normal `states_dot`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled states_dot at node node_index
        """
        self._nlp.states_dot.node_index = self.node_index

        out = self._nlp.states_dot.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def algebraic_states_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled algebraic_states associated with the current node index.

        Warning: Most of the time, the user does not want that algebraic_states variables but the normal
        `algebraic_states`, that said, it can sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled algebraic_states variables at node node_index
        """
        self._nlp.algebraic_states.node_index = self.node_index
        out = self._nlp.algebraic_states.scaled
        out.current_cx_to_get = self.cx_index_to_get
        return out

    @property
    def parameters(self) -> OptimizationVariableList:
        """
        Return the parameters

        Returns
        -------
        The parameters
        """
        return self._nlp.parameters

    @property
    def parameters_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled parameters

        Warning: Most of the time, the user does not want that parameters but the normal `parameters`, that said, it can
        sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled parameters
        """
        return MX() if type(self._nlp.parameters.scaled) == DM else self._nlp.parameters.scaled

    @property
    def q(self) -> CX:
        if "q" in self.states:
            return self.states["q"].mapping.to_second.map(self.states["q"].cx)
        elif "q_roots" in self.states and "q_joints" in self.states:
            # TODO: add mapping for q_roots and q_joints
            cx_start = vertcat(self.states["q_roots"].cx, self.states["q_joints"].cx)
            q_parent_list = OptimizationVariableList(
                self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
            )
            q_parent_list._cx_start = cx_start
            q = OptimizationVariable(
                name="q",
                cx_start=cx_start,
                index=[i for i in range(self.states["q_roots"].shape + self.states["q_joints"].shape)],
                mapping=BiMapping(
                    [i for i in range(self.states["q_roots"].shape + self.states["q_joints"].shape)],
                    [i for i in range(self.states["q_roots"].shape + self.states["q_joints"].shape)],
                ),
                parent_list=q_parent_list,
            )
            return q.cx
        else:
            raise RuntimeError("q is not defined in the states")

    @property
    def qdot(self) -> CX:
        if "qdot" in self.states:
            return self.states["qdot"].mapping.to_second.map(self.states["qdot"].cx)
        elif "qdot_roots" in self.states and "qdot_joints" in self.states:
            # TODO: add mapping for qdot_roots and qdot_joints
            cx_start = vertcat(self.states["qdot_roots"].cx_start, self.states["qdot_joints"].cx_start)
            qdot_parent_list = OptimizationVariableList(
                self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
            )
            qdot_parent_list._cx_start = cx_start
            qdot = OptimizationVariable(
                name="qdot",
                cx_start=cx_start,
                index=[i for i in range(self.states["qdot_roots"].shape + self.states["qdot_joints"].shape)],
                mapping=BiMapping(
                    [i for i in range(self.states["qdot_roots"].shape + self.states["qdot_joints"].shape)],
                    [i for i in range(self.states["qdot_roots"].shape + self.states["qdot_joints"].shape)],
                ),
                parent_list=qdot_parent_list,
            )
            return qdot.cx

    @property
    def tau(self) -> CX:
        if "tau" in self.controls:
            return self.controls["tau"].mapping.to_second.map(self.controls["tau"].cx)
        elif "tau_joints" in self.controls:
            return self.controls["tau_joints"].mapping.to_second.map(self.controls["tau_joints"].cx)

    @property
    def external_forces(self) -> CX:
        return self._nlp.get_external_forces(
            self.states.cx, self.controls.cx, self.algebraic_states.cx, self.numerical_timeseries.cx
        )

    def copy(self):
        return PenaltyController(
            self.ocp,
            self._nlp,
            self.t,
            self.x,
            self.u,
            self.x_scaled,
            self.u_scaled,
            self.p,
            self.a,
            self.a_scaled,
            self.node_index,
        )
