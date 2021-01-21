from ..limits.path_conditions import Bounds, InitialGuess
from .enums import ControlType, OdeSolver


class NonLinearProgram:
    """
    A nonlinear program that describe a phase in the ocp

    Attributes
    ----------
    casadi_func: dict
        All the declared casadi function
    contact_forces_func = function
        The contact force function if exists for the current nlp
    control_type: ControlType
        The control type for the current nlp
    CX: Union[MX, SX]
        The type of symbolic variable that is used
    dt: float
        The delta time of the current phase
    dynamics: list[ODE_SOLVER]
        All the dynamics for each of the node of the phase
    dynamics_func: function
        The dynamic function used during the current phase
    dynamics_type: Dynamics
        The dynamic option declared by the user for the current phase
    external_forces: list
        The external forces acting at the center of mass of the designated segment
    g: list[list[Constraint]]
        All the constraints at each of the node of the phase
    g_bounds: list
        All the constraint bounds of the constraints
    irk_polynomial_interpolation_degree: int
        The degree of the IRK  # TODO: these option should be from a special class
    J: list[list[Objective]]
        All the objectives at each of the node of the phase
    mapping: dict
        All the BidirectionalMapping of the states and controls
    model: biorbd.Model
        The biorbd model associated with the phase
    muscleNames: list[str]
        List of all the muscle names
    muscles: MX
        The casadi variables for the muscles
    nb_integration_steps: int
        The number of finite element of the RK  # TODO: these option should be from a special class
    nb_threads: int
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
    par_dynamics: casadi.Function
        The casadi function of the threaded dynamics
    parameters_to_optimize: dict
        The collection of parameters to optimize
    phase_idx: int
        The index of the current nlp in the ocp.nlp structure
    plot: dict
        The collection of plot for each of the variables
    q: MX
        The casadi variables for the generalized coordinates
    q_dot: MX
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
    """

    def __init__(self):
        self.casadi_func = {}
        self.contact_forces_func = None
        self.control_type = ControlType.NONE
        self.CX = None
        self.dt = None
        self.dynamics = []
        self.dynamics_func = None
        self.dynamics_type = None
        self.external_forces = []
        self.g = []
        self.g_bounds = []
        self.irk_polynomial_interpolation_degree = None
        self.J = []
        self.mapping = {}
        self.model = None
        self.muscleNames = None
        self.muscles = None
        self.nb_integration_steps = None
        self.nb_threads = None
        self.np = None
        self.ns = None
        self.nu = None
        self.nx = None
        self.ode_solver = OdeSolver.NO_SOLVER
        self.p = None
        self.par_dynamics = None
        self.parameters_to_optimize = {}
        self.phase_idx = None
        self.plot = {}
        self.q = None
        self.q_dot = None
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
