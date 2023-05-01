from typing import Callable, Any

import casadi
from casadi import SX, MX, Function, horzcat

from .optimization_variable import OptimizationVariableList, OptimizationVariable, OptimizationVariableContainer
from ..dynamics.ode_solver import OdeSolver
from ..limits.path_conditions import Bounds, InitialGuess, BoundsList
from ..misc.enums import ControlType
from ..misc.options import OptionList
from ..misc.mapping import NodeMapping
from ..dynamics.dynamics_evaluation import DynamicsEvaluation
from ..interfaces.biomodel import BioModel


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
    implicit_dynamics_func: Callable
        The implicit dynamic function used during the current phase f(x,u,p,xdot) = 0
    dynamics_type: Dynamics
        The dynamic option declared by the user for the current phase
    external_forces: list
        The external forces acting at the center of mass of the designated segment
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
    model: BiorbdModel | BioModel
        The biorbd model associated with the phase
    n_threads: int
        The number of thread to use
    ns: int
        The number of shooting points
    ode_solver: OdeSolver
        The chosen ode solver
    parameters: ParameterList
        Reference to the optimized parameters in the underlying ocp
    par_dynamics: casadi.Function
        The casadi function of the threaded dynamics
    phase_idx: int
        The index of the current nlp in the ocp.nlp structure
    plot: dict
        The collection of plot for each of the variables
    plot_mapping: list
        The mapping for the plots
    t0: float
        The time stamp of the beginning of the phase
    tf: float
        The time stamp of the end of the phase
    t_initial_guess: float
        The initial guess of the time
    variable_mappings: BiMappingList
        The list of mapping for all the variables
    u_bounds = Bounds()
        The bounds for the controls
    u_init = InitialGuess()
        The initial guess for the controls
    U: list[MX | SX]
        The casadi variables for the integration at each node of the phase
    controls: OptimizationVariableList
        A list of all the control variables
    x_bounds = Bounds()
        The bounds for the states
    x_init = InitialGuess()
        The initial guess for the states
    X: list[MX | SX]
        The casadi variables for the integration at each node of the phase
    states: OptimizationVariableList
        A list of all the state variables

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
    def add_casadi_func(self, name: str, function: Callable, *all_param: Any) -> casadi.Function:
        Add to the pool of declared casadi function. If the function already exists, it is skipped
    """

    def __init__(self):
        self.casadi_func = {}
        self.contact_forces_func = None
        self.soft_contact_forces_func = None
        self.control_type = ControlType.NONE
        self.cx = None
        self.dt = None
        self.dynamics = []
        self.dynamics_evaluation = DynamicsEvaluation()
        self.dynamics_func = None
        self.implicit_dynamics_func = None
        self.dynamics_type = None
        self.external_forces: list[Any] = []
        self.g = []
        self.g_internal = []
        self.g_implicit = []
        self.J = []
        self.J_internal = []
        self.model: BioModel | None = None
        self.n_threads = None
        self.ns = None
        self.ode_solver = OdeSolver.RK4()
        self.parameters = []
        self.par_dynamics = None
        self.phase_idx = None
        self.phase_mapping = None
        self.plot = {}
        self.plot_mapping = {}
        self.t0 = None
        self.tf = None
        self.t_initial_guess = None
        self.variable_mappings = {}
        self.u_bounds = Bounds()
        self.u_init = InitialGuess()
        self.U_scaled = None
        self.U = None
        self.use_states_from_phase_idx = NodeMapping()
        self.use_controls_from_phase_idx = NodeMapping()
        self.use_states_dot_from_phase_idx = NodeMapping()
        self.x_bounds = Bounds()
        self.x_init = InitialGuess()
        self.X_scaled = None
        self.X = None
        self.states = OptimizationVariableContainer()
        self.states_dot = OptimizationVariableContainer()
        self.controls = OptimizationVariableContainer()

    def initialize(self, cx: Callable = None):
        """
        Reset an nlp to a sane initial state

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

        self.states = self.states._set_states_and_controls(n_shooting=self.ns + 1, cx=self.cx)
        self.states_dot = self.states_dot._set_states_and_controls(n_shooting=self.ns + 1, cx=self.cx)
        self.controls = self.controls._set_states_and_controls(n_shooting=self.ns + 1, cx=self.cx)

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
                    f"{param_name} size({len(param)}) does not correspond " f"to the number of phases({ocp.n_phases})."
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

    @staticmethod
    def add_path_condition(ocp, var: Any, path_name: str, type_option: Any, type_list: Any):
        """
        Interface to add for PathCondition classes

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        var: Any
            The actual data to store
        path_name: str
            The name of the condition ("x_init", "u_bounds", ...)
        type_option: AnyOption
            The type of PathCondition
        type_list: AnyList
            The type of PathConditionList (must be the same type of path_type_option)
        """

        setattr(ocp, f"isdef_{path_name}", True)
        name = type_option.__name__
        if isinstance(var, type_option):
            var_tp = type_list()
            try:
                if isinstance(var_tp, BoundsList):
                    var_tp.add(bounds=var)
                else:
                    var_tp.add(var)
            except TypeError:
                raise RuntimeError(f"{path_name} should be built from a {name} or {name}List")
            var = var_tp
        elif not isinstance(var, type_list):
            raise RuntimeError(f"{path_name} should be built from a {name} or {name}List")
        NonLinearProgram.add(ocp, path_name, var, False)

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
    def mx_to_cx(name: str, symbolic_expression: SX | MX | Callable, *all_param: Any) -> Function:
        """
        Add to the pool of declared casadi function. If the function already exists, it is skipped

        Parameters
        ----------
        name: str
            The unique name of the function to add to the casadi functions pool
        symbolic_expression: SX | MX | Callable
            The symbolic expression to be converted, also support Callables
        all_param: Any
            Any parameters to pass to the biorbd function
        """

        from ..optimization.optimization_variable import OptimizationVariable, OptimizationVariableList
        from ..optimization.parameters import Parameter, ParameterList

        cx_types = OptimizationVariable, OptimizationVariableList, Parameter, ParameterList
        mx = [var.mx if isinstance(var, cx_types) else var for var in all_param]
        cx = [
            var.mapping.to_second.map(var.cx_start) if hasattr(var, "mapping") else var.cx_start
            for var in all_param
            if isinstance(var, cx_types)
        ]
        return NonLinearProgram.to_casadi_func(name, symbolic_expression, *mx)(*cx)

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
        return func.expand() if expand else func

    def time(self, node_idx: int):
        """
        Gives the time in the current nlp

        Parameters
        ----------
        node_idx: int
          number of the node

        Returns
        -------
        The time in the current nlp
        """
        if node_idx < 0 or node_idx > self.ns:
            return ValueError("Node_number out of range [0:nlp.ns]")
        return self.tf / self.ns * node_idx
