from typing import Callable, Any

import casadi
from casadi import SX, MX, Function, horzcat, vertcat

from .optimization_variable import OptimizationVariable, OptimizationVariableContainer
from ..dynamics.ode_solver import OdeSolver
from ..limits.path_conditions import InitialGuessList, BoundsList
from ..misc.enums import ControlType, PhaseDynamics
from ..misc.options import OptionList
from ..misc.mapping import NodeMapping
from ..dynamics.dynamics_evaluation import DynamicsEvaluation
from ..dynamics.dynamics_functions import DynamicsFunctions
from ..models.protocols.biomodel import BioModel
from ..models.protocols.holonomic_biomodel import HolonomicBioModel
from ..models.protocols.variational_biomodel import VariationalBioModel
from ..models.protocols.stochastic_biomodel import StochasticBioModel


class NonLinearProgram:
    """
    A nonlinear program that describes a phase in the ocp

    Attributes
    ----------
    casadi_func: dict
        All the declared casadi function
    contact_forces_func = function
        The contact force function if exists for the current nlp
    control_type: ControlType
        The control type for the current nlp
    cx: MX | SX
        The type of symbolic variable that is used
    dt: float
        The delta time of the current phase
    dynamics: list[ODE_SOLVER]
        All the dynamics for each of the node of the phase
    dynamics_evaluation: DynamicsEvaluation
        The dynamic MX or SX used during the current phase
    dynamics_func: Callable
        The dynamic function used during the current phase dxdt = f(x,u,p)
    extra_dynamics_func: Callable
        The extra dynamic function used during the current phase dxdt = f(x,u,p)
    implicit_dynamics_func: Callable
        The implicit dynamic function used during the current phase f(x,u,p,xdot) = 0
    dynamics_type: Dynamics
        The dynamic option declared by the user for the current phase
    g: list[list[Constraint]]
        All the constraints at each of the node of the phase
    g_internal: list[list[Constraint]]
        All the constraints internally defined by the OCP at each of the node of the phase
    g_implicit: list[list[Constraint]]
        All the implicit constraints defined by the OCP at each of the node of the phase
    J: list[list[Objective]]
        All the objectives at each of the node of the phase
    J_internal: list[list[Objective]]
        All the objectives internally defined by the phase at each of the node of the phase
    model: BiorbdModel | BioModel | StochasticBioModel | HolonomicBioModel | VariationalBioModel
        The biorbd model associated with the phase
    n_threads: int
        The number of thread to use
    ns: int
        The number of shooting points
    ode_solver: OdeSolverBase
        The chosen ode solver
    parameters: ParameterContainer
        Reference to the optimized parameters in the underlying ocp
    par_dynamics: casadi.Function
        The casadi function of the threaded dynamics
    phase_idx: int
        The index of the current nlp in the ocp.nlp structure
    plot: dict
        The collection of plot for each of the variables
    plot_mapping: list
        The mapping for the plots
    tf: float
        The time stamp of the end of the phase
    tf_mx:
        The time stamp of the end of the phase
    variable_mappings: BiMappingList
        The list of mapping for all the variables
    u_bounds = Bounds()
        The bounds for the controls
    u_init = InitialGuess()
        The initial guess for the controls
    u_scaling:
        The scaling for the controls
    U: list[MX | SX]
        The casadi variables for the integration at each node of the phase
    controls: OptimizationVariableContainer
        A list of all the control variables
    x_bounds = Bounds()
        The bounds for the states
    x_init = InitialGuess()
        The initial guess for the states
    X: list[MX | SX]
        The casadi variables for the integration at each node of the phase
    x_scaling:
        The scaling for the states
    states: OptimizationVariableContainer
        A list of all the state variables
    a_bounds = Bounds()
        The bounds for the algebraic_states variables
    a_init = InitialGuess()
        The initial guess for the algebraic_states variables
    a_scaling:
        The scaling for the algebraic_states variables
    phase_dynamics: PhaseDynamics
        The dynamics of the current phase (e.g. SHARED_DURING_PHASE, or ONE_PER_NODE)
    A: list[MX | SX]
        The casadi variables for the algebraic_states variables


    Methods
    -------
    initialize(self, cx: [MX, SX])
        Reset an nlp to a sane initial state
    add(ocp: OptimalControlProgram, param_name: str, param: Any, duplicate_singleton: bool,
            _type: Any = None, name: str = None)
        Set a parameter to their respective nlp
    __setattr(nlp, name: str | None, param_name: str, param: Any)
        Add a new element to the nlp of the format 'nlp.param_name = param' or 'nlp.name["param_name"] = param'
    add_path_condition(ocp: OptimalControlProgram, var: Any, path_name: str, type_option: Any, type_list: Any)
        Interface to add for PathCondition classes
    add_casadi_func(self, name: str, function: Callable, *all_param: Any) -> casadi.Function:
        Add to the pool of declared casadi function. If the function already exists, it is skipped
    to_casadi_func
        Converts a symbolic expression into a casadi function
    node_time(self, node_idx: int)
        Gives the time for a specific index
    """

    def __init__(self, phase_dynamics: PhaseDynamics, use_sx: bool):
        self.casadi_func = {}
        self.contact_forces_func = None
        self.soft_contact_forces_func = None
        self.control_type = ControlType.CONSTANT
        self.cx = None
        self.dt = None
        self.dynamics = []
        self.extra_dynamics = []
        self.dynamics_evaluation = DynamicsEvaluation()
        self.dynamics_func = None
        self.extra_dynamics_func: list = []
        self.implicit_dynamics_func = None
        self.dynamics_type = None
        self.numerical_timeseries = None
        self.g = []
        self.g_internal = []
        self.g_implicit = []
        self.J = []
        self.J_internal = []
        self.model: BioModel | StochasticBioModel | HolonomicBioModel | VariationalBioModel | None = None
        self.n_threads = None
        self.ns = None
        self.ode_solver = OdeSolver.RK4()
        self.par_dynamics = None
        self.phase_idx = None
        self.phase_mapping = None
        self.plot = {}
        self.plot_mapping = {}
        self.T = None
        self.variable_mappings = {}
        self.u_bounds = BoundsList()
        self.u_init = InitialGuessList()
        self.U_scaled = None
        self.u_scaling = None
        self.U = None
        self.use_states_from_phase_idx = NodeMapping()
        self.use_controls_from_phase_idx = NodeMapping()
        self.use_states_dot_from_phase_idx = NodeMapping()
        self.x_bounds = BoundsList()
        self.x_init = InitialGuessList()
        self.X_scaled = None
        self.x_scaling = None
        self.X = None
        self.a_bounds = BoundsList()
        self.a_init = InitialGuessList()
        self.A = None
        self.A_scaled = None
        self.a_scaling = None
        self.phase_dynamics = phase_dynamics
        self.time_index = None
        self.time_cx = None
        self.time_mx = None
        self.dt = None
        self.dt_mx = None
        self.tf = None
        self.tf_mx = None
        self.states = OptimizationVariableContainer(self.phase_dynamics)
        self.states_dot = OptimizationVariableContainer(self.phase_dynamics)
        self.controls = OptimizationVariableContainer(self.phase_dynamics)
        # parameters is currently a clone of ocp.parameters, but should hold phase parameters
        from ..optimization.parameters import ParameterContainer

        self.parameters = ParameterContainer(use_sx=use_sx)
        parameters_except_time = ParameterContainer(use_sx=use_sx)
        self.algebraic_states = OptimizationVariableContainer(self.phase_dynamics)
        self.integrated_values = OptimizationVariableContainer(self.phase_dynamics)
        self.numerical_timeseries = OptimizationVariableContainer(self.phase_dynamics)

    def initialize(self, cx: MX | SX | Callable = None):
        """
        Reset a nlp to a sane initial state

        Parameters
        ----------
        cx: MX | SX
            The type of casadi variable

        """

        self.plot = {}
        self.cx = cx
        self.J = []
        self.g = []
        self.g_internal = []
        self.casadi_func = {}
        self.states.initialize_from_shooting(n_shooting=self.ns + 1, cx=self.cx)
        self.states_dot.initialize_from_shooting(n_shooting=self.ns + 1, cx=self.cx)
        self.controls.initialize_from_shooting(n_shooting=self.ns + 1, cx=self.cx)
        self.algebraic_states.initialize_from_shooting(n_shooting=self.ns + 1, cx=self.cx)
        self.integrated_values.initialize_from_shooting(n_shooting=self.ns + 1, cx=self.cx)

    @property
    def n_states_nodes(self) -> int:
        """
        Returns
        -------
        The number of states
        """
        return self.ns + 1

    def n_states_decision_steps(self, node_idx) -> int:
        """
        Parameters
        ----------
        node_idx: int
            The index of the node

        Returns
        -------
        The number of states
        """
        if node_idx >= self.ns:
            return 1
        return self.dynamics[node_idx].shape_xf[1] + (1 if self.ode_solver.duplicate_starting_point else 0)

    def n_states_stepwise_steps(self, node_idx) -> int:
        """
        Parameters
        ----------
        node_idx: int
            The index of the node

        Returns
        -------
        The number of states
        """
        if node_idx >= self.ns:
            return 1
        if self.ode_solver.is_direct_collocation:
            return self.dynamics[node_idx].shape_xall[1] - (1 if not self.ode_solver.duplicate_starting_point else 0)
        else:
            return self.dynamics[node_idx].shape_xall[1]

    @property
    def n_controls_nodes(self) -> int:
        """
        Returns
        -------
        The number of controls
        """
        mod = 1 if self.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE) else 0
        return self.ns + mod

    def n_controls_steps(self, node_idx) -> int:
        """
        Parameters
        ----------
        node_idx: int
            The index of the node

        Returns
        -------
        The number of states
        """

        if self.control_type == ControlType.CONSTANT:
            return 1
        elif self.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            return 1
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            return 2
        else:
            raise RuntimeError("Not implemented yet")

    @property
    def n_algebraic_states_nodes(self) -> int:
        """
        Returns
        -------
        The number of controls
        """

        return self.n_states_nodes

    @staticmethod
    def n_algebraic_states_decision_steps(node_idx) -> int:
        """
        Parameters
        ----------
        node_idx: int
            The index of the node

        Returns
        -------
        The number of states
        """

        return 1

    @staticmethod
    def add(ocp, param_name: str, param: Any, duplicate_singleton: bool, _type: Any = None, name: str = None):
        """
        Set a parameter to their respective nlp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp where all the nlp are stored
        param_name: str
            The name of the parameter as it is named in the nlp structure, if name is not set.
            Otherwise, the param_name is store in nlp.name[param_name]
        param: Any
            The value of the parameter
        duplicate_singleton: bool
            If the value is unique, should this value be spread to all nlp (or raises an error)
        _type: Any
            The type of the data
        name: str
            The name of the parameter if the param_name should be stored in a dictionary
        """

        if isinstance(param, (OptionList, list, tuple)):
            if len(param) == 1 and ocp.n_phases != 1 and not duplicate_singleton:
                raise RuntimeError(
                    f"{param_name} size({len(param)}) does not correspond " f"to the number of phases({ocp.n_phases})."
                )

            for i in range(ocp.n_phases):
                cmp = 0 if len(param) == 1 else i
                NonLinearProgram.__setattr(ocp.nlp[i], name, param_name, param[cmp])

        else:
            if ocp.n_phases != 1 and not duplicate_singleton:
                raise RuntimeError(
                    f"{param_name} size({1 if isinstance(param, int) else len(param)}) does not correspond "
                    f"to the number of phases({ocp.n_phases})."
                    f"List length of model, final time and node shooting must be equivalent to phase number"
                )

            for i in range(ocp.n_phases):
                NonLinearProgram.__setattr(ocp.nlp[i], name, param_name, param)

        if _type is not None:
            for nlp in ocp.nlp:
                if (
                    (
                        (name is None and getattr(nlp, param_name) is not None)
                        or (name is not None and param is not None)
                    )
                    and not isinstance(param, _type)
                    and isinstance(param, (list, tuple))
                    and False in [False for i in param if not isinstance(i, _type)]
                ):
                    raise RuntimeError(f"Parameter {param_name} must be a {str(_type)}")

    @staticmethod
    def __setattr(nlp, name: str | None, param_name: str, param: Any):
        """
        Add a new element to the nlp of the format 'nlp.param_name = param' or 'nlp.name["param_name"] = param'

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to a nlp
        name: str | None
            A meta name of the param if the param is set in a dictionary
        param_name: str
            The name of the parameter
        param: Any
            The parameter itself
        """

        if name is None:
            setattr(nlp, param_name, param)
        else:
            getattr(nlp, name)[param_name] = param

    def add_casadi_func(self, name: str, function: Callable | SX | MX, *all_param: Any) -> casadi.Function:
        """
        Add to the pool of declared casadi function. If the function already exists, it is skipped

        Parameters
        ----------
        name: str
            The unique name of the function to add to the casadi functions pool
        function: Callable | SX | MX
            The biorbd function to add
        all_param: dict
            Any parameters to pass to the biorbd function
        """

        if name in self.casadi_func:
            return self.casadi_func[name]
        else:
            mx = [var.mx if isinstance(var, OptimizationVariable) else var for var in all_param]
            self.casadi_func[name] = self.to_casadi_func(name, function, *mx)
        return self.casadi_func[name]

    @staticmethod
    def to_casadi_func(name, symbolic_expression: MX | SX | Callable, *all_param, expand=True) -> Function:
        """
        Converts a symbolic expression into a casadi function

        Parameters
        ----------
        name: str
            The name of the function
        symbolic_expression: MX | SX | Callable
            The symbolic expression to be converted, also support Callables
        all_param: Any
            Any parameters to pass to the biorbd function
        expand: bool
            If the function should be expanded

        Returns
        -------
        The converted function

        """
        cx_param = []
        for p in all_param:
            if isinstance(p, (MX, SX)):
                cx_param.append(p)

        if isinstance(symbolic_expression, (MX, SX, Function)):
            func_evaluated = symbolic_expression
        else:
            func_evaluated = symbolic_expression(*all_param)
            if isinstance(func_evaluated, (list, tuple)):
                func_evaluated = horzcat(*[val if isinstance(val, MX) else val.to_mx() for val in func_evaluated])
            elif not isinstance(func_evaluated, MX):
                func_evaluated = func_evaluated.to_mx()
        func = Function(name, cx_param, [func_evaluated])

        if expand:
            try:
                func = func.expand()
            except Exception as me:
                raise RuntimeError(
                    f"An error occurred while executing the 'expand()' function for {name}. Please review the following "
                    "casadi error message for more details.\n"
                    "Several factors could be causing this issue. If you are creating your own casadi function, "
                    "it is possible that you have free variables. Another possibility, if you are using a predefined "
                    "function, the error might be due to the inability to use expand=True at all. In that case, try "
                    "adding expand=False to the dynamics or the penalty.\n"
                    "Original casadi error message:\n"
                    f"{me}"
                )

        return func.expand() if expand else func

    def node_time(self, node_idx: int):
        """
        Gives the time for a specific index

        Parameters
        ----------
        node_idx: int
          Index of the node

        Returns
        -------
        The time for a specific index
        """
        if node_idx < 0 or node_idx > self.ns:
            return ValueError(f"node_index out of range [0:{self.ns}]")
        return self.tf / self.ns * node_idx

    def get_var_from_states_or_controls(
        self, key: str, states: MX.sym, controls: MX.sym, algebraic_states: MX.sym = None
    ) -> MX:
        """
        This function returns the requested variable from the states, controls, or algebraic_states.
        If the variable is present in more than one type of variables, it returns the following priority:
        1) states
        2) controls
        3) algebraic_states
        If the variable is split in its roots and joints components, it returns the concatenation of for the states,
        and only the joints for the controls.
        Please note that this function is not meant to be used by the user directly, but should be an internal function.

        Parameters
        ----------
        key: str
            The name of the variable to return
        states: MX.sym
            The states
        controls: MX.sym
            The controls
        algebraic_states: MX.sym
            The algebraic_states
        """
        if key in self.states:
            out_nlp, out_var = (self.states[key], states)
            out = DynamicsFunctions.get(out_nlp, out_var)
        elif f"{key}_roots" in self.states and f"{key}_joints" in self.states:
            out_roots_nlp, out_roots_var = (self.states[f"{key}_roots"], states)
            out_roots = DynamicsFunctions.get(out_roots_nlp, out_roots_var)
            out_joints_nlp, out_joints_var = (self.states[f"{key}_joints"], states)
            out_joints = DynamicsFunctions.get(out_joints_nlp, out_joints_var)
            out = vertcat(out_roots, out_joints)
        elif key in self.controls:
            out_nlp, out_var = (self.controls[key], controls)
            out = DynamicsFunctions.get(out_nlp, out_var)
        elif f"{key}_joints" in self.controls:
            out_joints_nlp, out_joints_var = (self.controls[f"{key}_joints"], controls)
            out = DynamicsFunctions.get(out_joints_nlp, out_joints_var)
        elif key in self.algebraic_states:
            out_nlp, out_var = (self.algebraic_states[key], algebraic_states)
            out = DynamicsFunctions.get(out_nlp, out_var)
        else:
            raise RuntimeError(f"{key} not found in states or controls")
        return out
