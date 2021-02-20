from typing import Callable, Any, Union

from .parameters import ParameterList
from ..dynamics.ode_solver import OdeSolver
from ..limits.path_conditions import Bounds, InitialGuess, BoundsList
from ..misc.enums import ControlType
from ..misc.options import OptionList


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
    cx: Union[MX, SX]
        The type of symbolic variable that is used
    dt: float
        The delta time of the current phase
    dynamics: list[ODE_SOLVER]
        All the dynamics for each of the node of the phase
    dynamics_func: Callable
        The dynamic function used during the current phase
    dynamics_type: Dynamics
        The dynamic option declared by the user for the current phase
    external_forces: list
        The external forces acting at the center of mass of the designated segment
    g: list[list[Constraint]]
        All the constraints at each of the node of the phase
    J: list[list[Objective]]
        All the objectives at each of the node of the phase
    mapping: dict
        All the BiMapping of the states and controls
    model: biorbd.Model
        The biorbd model associated with the phase
    muscleNames: list[str]
        List of all the muscle names
    muscles: MX
        The casadi variables for the muscles
    n_threads: int
        The number of thread to use
    np: int
        The number of parameters
    ns: int
        The number of shooting points
    nu: int
        The number of controls
    nx: int
        The number of states
    ode_solver: OdeSolver
        The chosen ode solver
    p: MX
        The casadi variables for the parameters
    parameters: ParameterList
        Reference to the optimized parameters in the underlying ocp
    par_dynamics: casadi.Function
        The casadi function of the threaded dynamics
    phase_idx: int
        The index of the current nlp in the ocp.nlp structure
    plot: dict
        The collection of plot for each of the variables
    q: MX
        The casadi variables for the generalized coordinates
    qdot: MX
        The casadi variables for the generalized velocities
    shape: dict
        A collection of the dimension of each of the variables
    tau: MX
        The casadi variables for the generalized torques
    t0: float
        The time stamp of the beginning of the phase
    tf: float
        The time stamp of the end of the phase
    u: MX
        The casadi variables for the controls
    U: list[Union[MX, SX]]
        The casadi variables for the integration at each node of the phase
    u_bounds = Bounds()
        The bounds for the controls
    u_init = InitialGuess()
        The initial guess for the controls
    var_controls: dict
        The number of elements for each control the key is the name of the control
    var_states: dict
        The number of elements for each state the key is the name of the state
    x: MX
        The casadi variables for the states
    X: list[Union[MX, SX]]
        The casadi variables for the integration at each node of the phase
    x_bounds = Bounds()
        The bounds for the states
    x_init = InitialGuess()
        The initial guess for the states

    Methods
    -------
    initialize(self, cx: [MX, SX])
        Reset an nlp to a sane initial state
    add(ocp: OptimalControlProgram, param_name: str, param: Any, duplicate_singleton: bool,
            _type: Any = None, name: str = None)
        Set a parameter to their respective nlp
    add_path_condition(ocp: OptimalControlProgram, var: Any, path_name: str, type_option: Any, type_list: Any)
        Interface to add for PathCondition classes
    """

    def __init__(self):
        self.casadi_func = {}
        self.contact_forces_func = None
        self.control_type = ControlType.NONE
        self.cx = None
        self.dt = None
        self.dynamics = []
        self.dynamics_func = None
        self.dynamics_type = None
        self.external_forces = []
        self.g = []
        self.J = []
        self.mapping = {}
        self.model = None
        self.muscleNames = None
        self.muscles = None
        self.n_threads = None
        self.np = None
        self.ns = None
        self.nu = None
        self.nx = None
        self.ode_solver = OdeSolver.RK4()
        self.p = None
        self.parameters = ParameterList()
        self.par_dynamics = None
        self.phase_idx = None
        self.plot = {}
        self.q = None
        self.qdot = None
        self.shape = {}
        self.tau = None
        self.t0 = None
        self.tf = None
        self.u = None
        self.U = None
        self.u_bounds = Bounds()
        self.u_init = InitialGuess()
        self.var_controls = {}
        self.var_states = {}
        self.x = None
        self.X = None
        self.x_bounds = Bounds()
        self.x_init = InitialGuess()

    def initialize(self, cx: Callable = None):
        """
        Reset an nlp to a sane initial state

        Parameters
        ----------
        cx: Union[MX, SX]
            The type of casadi variable

        """
        self.shape = {"q": 0, "qdot": 0, "tau": 0, "muscle": 0}
        self.plot = {}
        self.var_states = {}
        self.var_controls = {}
        self.cx = cx
        self.x = self.cx()
        self.u = self.cx()
        self.J = []
        self.g = []
        self.casadi_func = {}

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
                    f"{param_name} size({len(param)}) does not correspond "
                    f"to the number of phases({ocp.n_phases})."
                )

            for i in range(ocp.n_phases):
                cmp = 0 if len(param) == 1 else i
                NonLinearProgram.__setattr(ocp.nlp[i], name, param_name, param[cmp])

        else:
            if ocp.n_phases != 1 and not duplicate_singleton:
                raise RuntimeError(
                    f"{param_name} size({len(param)}) does not correspond "
                    f"to the number of phases({ocp.n_phases})."
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
    def __setattr(nlp, name: Union[str, None], param_name: str, param: Any):
        """
        Add a new element to the nlp of the format 'nlp.param_name = param' or 'nlp.name["param_name"] = param'
        
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to a nlp
        name: Union[str, None]
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
