class NonLinearProgram:
    def __init__(
            self,
            model,
            phase_idx,
            ns,
            tf,
            t0,
            dt,
            nb_threads,
            external_forces,
            X_bounds,
            U_bounds,
            dynamics_type,
            ode_solver,
            control_type,
            X_init,
            U_init,
            nb_integration_steps,
            plot,
            var_states,
            var_controls,
            CX,
            x,
            u,
            J,
            g,
            g_bounds,
            casadi_func,
            dynamics,
            par_dynamics,
            dynamics_func,
            nx,
            nu,
            X,
            U,
            p,
            size={},
            mapping={},
            ):
        self.model = model
        self.phase_idx = phase_idx
        self.ns = ns
        self.tf = tf
        self.t0 = t0
        self.dt = dt
        self.nb_threads = nb_threads
        self.external_forces = external_forces
        self.X_bounds = X_bounds
        self.U_bounds = U_bounds
        self.dynamics_type = dynamics_type
        self.ode_solver = ode_solver
        self.control_type = control_type
        self.X_init = X_init
        self.U_init = U_init
        self.nb_integration_steps = nb_integration_steps
        self.plot = plot
        self.var_states = var_states
        self.var_controls = var_controls
        self.CX = CX
        self.x = x
        self.u = u
        self.J = J
        self.g = g
        self.g_bounds = g_bounds
        self.casadi_func = casadi_func
        self.dynamics = dynamics
        self.par_dynamics = par_dynamics
        self.dynamics_func = dynamics_func
        self.nx = nx
        self.nu = nu
        self.X = X
        self.U = U
        self.p = p
        self.size = size,
        self.mapping = mapping,
