from casadi import MX

from ..limits.path_conditions import Bounds, InitialGuess
from .. import ControlType, OdeSolver, DynamicsFcn


class NonLinearProgram:
    def __init__(
        self,
        CX=MX,
        J=[],
        U=[],
        u_bounds=Bounds(),
        u_init=InitialGuess(),
        X=[],
        x_bounds=Bounds(),
        x_init=InitialGuess(),
        casadi_func={},
        contact_forces_func=None,
        control_type=ControlType.CONSTANT,
        dt=0.0,
        dynamics=[],
        dynamics_func=None,
        dynamics_type=DynamicsFcn.TORQUE_DRIVEN,
        external_forces=None,
        g=[],
        g_bounds=Bounds(),
        mapping={},
        model=None,
        muscleNames=[],
        muscles=None,
        nb_integration_steps=0,
        nb_threads=1,
        np=None,
        ns=0,
        nu=0,
        nx=0,
        ode_solver=OdeSolver.RK4,
        p=None,
        par_dynamics={},
        parameters_to_optimize={},
        phase_idx=0,
        plot={},
        problem_type={},
        q=None,
        q_dot=None,
        shape={},
        tau=None,
        t0=0.0,
        tf=0.0,
        u=MX(),
        var_controls={},
        var_states={},
        x=MX(),
    ):
        self.CX = CX
        self.J = J
        self.U = U
        self.u_bounds = u_bounds
        self.u_init = u_init
        self.X = X
        self.x_bounds = x_bounds
        self.x_init = x_init
        self.casadi_func = casadi_func
        self.contact_forces_func = contact_forces_func
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
        self.muscleNames = muscleNames
        self.muscles = muscles
        self.nb_integration_steps = nb_integration_steps
        self.nb_threads = nb_threads
        self.np = np
        self.ns = ns
        self.nu = nu
        self.nx = nx
        self.ode_solver = ode_solver
        self.p = p
        self.par_dynamics = par_dynamics
        self.parameters_to_optimize = parameters_to_optimize
        self.phase_idx = phase_idx
        self.plot = plot
        self.problem_type = problem_type
        self.q = q
        self.q_dot = q_dot
        self.shape = shape
        self.tau = tau
        self.t0 = t0
        self.tf = tf
        self.u = u
        self.var_controls = var_controls
        self.var_states = var_states
        self.x = x
