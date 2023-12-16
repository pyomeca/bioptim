from typing import Any, Callable

from casadi import MX, SX, vertcat

from ..dynamics.ode_solver import OdeSolver
from ..misc.enums import ControlType, PhaseDynamics
from ..misc.mapping import BiMapping
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import OptimizationVariableList, OptimizationVariable


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
        t: list,
        x: list,
        u: list,
        x_scaled: list,
        u_scaled: list,
        p: MX | SX | list,
        s: list,
        s_scaled: list,
        node_index: int = None,
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
        s: list
            References to the stochastic variables
        s_scaled: list
            References to the scaled stochastic variables
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
        self.s = s
        self.s_scaled = s_scaled
        self.p = vertcat(p) if p is not None else p
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
    def t_span(self) -> list:
        dt = self.phases_time_cx[self.phase_idx]
        return vertcat(self.time_cx, self.time_cx + dt) + self.node_index * dt

    @property
    def phases_time_cx(self) -> list:
        return self.ocp.dt_parameter.cx

    @property
    def time_cx(self) -> MX | SX | Callable:
        return self._nlp.time_cx

    @property
    def cx(self) -> MX | SX | Callable:
        return self._nlp.cx

    @property
    def to_casadi_func(self) -> Callable:
        return self._nlp.to_casadi_func

    @property
    def control_type(self) -> ControlType:
        return self._nlp.control_type

    @property
    def ode_solver(self) -> OdeSolver:
        return self._nlp.ode_solver

    @property
    def phase_idx(self) -> int:
        return self._nlp.phase_idx

    @property
    def ns(self) -> int:
        return self._nlp.ns

    @property
    def mx_to_cx(self):
        return self._nlp.mx_to_cx

    @property
    def model(self):
        return self._nlp.model

    @property
    def dt(self) -> MX | SX:
        return self._nlp.dt

    @property
    def tf(self) -> MX | SX:
        return self._nlp.tf
    
    @property
    def time(self) -> OptimizationVariable:
        """
        Return the time associated with the current node index

        Returns
        -------
        The time at node node_index
        """

        tp = OptimizationVariableList(self._nlp.cx, self._nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE)
        tp.append(
            "time",
            mx=self._nlp.time_mx,
            cx=[self._nlp.time_cx, self._nlp.time_cx, self._nlp.time_cx],
            bimapping=BiMapping(to_second=[0], to_first=[0]),
        )
        return tp["time"]

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
    def stochastic_variables(self) -> OptimizationVariableList:
        """
        Return the stochastic_variables associated with the current node index
        Returns
        -------
        The stochastic_variables at node node_index
        """
        # TODO: This variables should be renamed to "algebraic"
        self._nlp.stochastic_variables.node_index = self.node_index
        out = self._nlp.stochastic_variables.unscaled
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
    def integrate(self):
        return self._nlp.dynamics[self.node_index]

    def integrate_extra_dynamics(self, dynamics_index):
        return self._nlp.extra_dynamics[dynamics_index][self.node_index]

    @property
    def dynamics(self):
        return self._nlp.dynamics_func[0]

    def extra_dynamics(self, dynamics_index):
        # +1 - index so "integrate_extra_dynamics" and "extra_dynamics" share the same index.
        # This is a hack which should be dealt properly at some point
        return self._nlp.dynamics_func[dynamics_index + 1]

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
    def stochastic_variables_scaled(self) -> OptimizationVariableList:
        """
        Return the scaled stochastic variables associated with the current node index.

        Warning: Most of the time, the user does not want that stochastic variables but the normal
        `stochastic_variables`, that said, it can sometime be useful for very limited number of use case.

        Returns
        -------
        The scaled stochastic variables at node node_index
        """
        self._nlp.stochastic_variables.node_index = self.node_index
        out = self._nlp.stochastic_variables.scaled
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
        return self._nlp.parameters.scaled

    def get_time_parameter_idx(self):
        time_idx = None
        for i in range(self.parameters.cx.shape[0]):
            param_name = self.parameters.cx[i].name()
            if param_name == "time_phase_" + str(self.phase_idx):
                time_idx = self.phase_idx
        if time_idx is None:
            raise RuntimeError(
                f"Time penalty can't be established since the {self.phase_idx}th phase has no time parameter. "
                f"\nTime parameter can be added with : "
                f"\nobjective_functions.add(ObjectiveFcn.[Mayer or Lagrange].MINIMIZE_TIME) or "
                f"\nwith constraints.add(ConstraintFcn.TIME_CONSTRAINT)."
            )
        return time_idx

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
            self.s,
            self.s_scaled,
            self.node_index,
        )
