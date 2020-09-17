from casadi import MX

from ..limits.path_conditions import Bounds, InitialConditions
from .. import ControlType, OdeSolver, DynamicsType


class NonLinearProgram:
    def __init__(
        self,
        CX=MX,
        J=[],
        U=[],
        U_bounds=Bounds(),
        U_init=InitialConditions(),
        X=[],
        X_bounds=Bounds(),
        X_init=InitialConditions(),
        casadi_func={},
        control_type=ControlType.CONSTANT,
        dt=0.,
        dynamics=[],
        dynamics_func=None,
        dynamics_type=DynamicsType.TORQUE_DRIVEN,
        external_forces=None,
        g=[],
        g_bounds=Bounds(),
        mapping={},
        model=None,
        nb_integration_steps=0,
        nb_threads=1,
        ns=0,
        nu=0,
        nx=0,
        ode_solver=OdeSolver.RK,
        p=None,
        par_dynamics={},
        parameters_to_optimize={},
        phase_idx=0,
        plot={},
        shape={},
        t0=0.,
        tf=0.,
        u=MX(),
        var_controls={},
        var_states={},
        x=MX(),
    ):
        self.CX = CX
        self.J = J
        self.U = U
        self.U_bounds = U_bounds
        self.U_init = U_init
        self.X = X
        self.X_bounds = X_bounds
        self.X_init = X_init
        self.casadi_func = casadi_func
        self.control_type = control_type
        self.dt = dt
        self.dynamics = dynamics
        self.dynamics_func = dynamics_func
        self.dynamics_type = dynamics_type
        self.external_forces = external_forces
        self.g = g
        self.g_bounds = g_bounds
        self.mapping = mapping
        self.model = model
        self.nb_integration_steps = nb_integration_steps
        self.nb_threads = nb_threads
        self.ns = ns
        self.nu = nu
        self.nx = nx
        self.ode_solver = ode_solver
        self.p = p
        self.par_dynamics = par_dynamics
        self.parameters_to_optimize = parameters_to_optimize
        self.phase_idx = phase_idx
        self.plot = plot
        self.shape = shape
        self.t0 = t0
        self.tf = tf
        self.u = u
        self.var_controls = var_controls
        self.var_states = var_states
        self.x = x
