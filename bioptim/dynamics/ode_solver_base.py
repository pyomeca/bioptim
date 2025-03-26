from typing import Callable

from casadi import MX, SX, integrator as casadi_integrator, horzcat, Function, vertcat

from . import integrator
from ..misc.enums import ControlType, DefectType, PhaseDynamics


class OdeSolverBase:
    """
    The base class for the ODE solvers

    Methods
    -------
    integrator(self, ocp, nlp, node_index) -> list
        The interface of the OdeSolver to the corresponding integrator
    prepare_dynamic_integrator(ocp, nlp)
        Properly set the integration in an nlp
    """

    def __init__(self, duplicate_starting_point: bool = False):
        """
        Parameters
        ----------
        duplicate_starting_point: bool
            If the starting point should be duplicated in the integrator's casadi function, mostly used in Stochastic OCPs
        """

        self.duplicate_starting_point = duplicate_starting_point

    @property
    def integrator(self):
        """
        The corresponding integrator class

        Returns
        -------
        The integrator class
        """
        raise RuntimeError("This method should be implemented in the child class")

    @property
    def is_direct_collocation(self) -> bool:
        """
        indicating if the ode solver is direct collocation method

        Returns
        -------
        True if the ode solver is direct collocation method
        """
        raise RuntimeError("This method should be implemented in the child class")

    @property
    def is_direct_shooting(self) -> bool:
        """
        indicating if the ode solver is direct shooting method

        Returns
        -------
        True if the ode solver is direct shooting method
        """
        raise RuntimeError("This method should be implemented in the child class")

    @property
    def n_required_cx(self) -> int:
        """
        The required number of column required in the casadi CX matrix for the state variables

        Returns
        -------
        The number of required casadi functions
        """
        raise RuntimeError("This method should be implemented in the child class")

    @property
    def defects_type(self) -> DefectType:
        """
        The type of defect

        Returns
        -------
        The type of defect
        """
        raise RuntimeError("This method should be implemented in the child class")

    def t_ode(self, nlp) -> list:
        """
        The time span of the integration

        Parameters
        ----------
        nlp
            The NonLinearProgram handler

        Returns
        -------
        The time span of the integration
        """
        return vertcat(nlp.time_cx, nlp.dt)

    def x_ode(self, nlp) -> MX:
        """
        The symbolic state variables

        Parameters
        ----------
        nlp
            The NonLinearProgram handler

        Returns
        -------
        The symbolic state variables
        """
        raise RuntimeError("This method should be implemented in the child class")

    def p_ode(self, nlp) -> MX:
        """
        The symbolic controls. The nomenclature is p_ode (instead of the intuitive u_ode) to be consistent with
        the scipy integrator

        Parameters
        ----------
        nlp
            The NonLinearProgram handler

        Returns
        -------
        The symbolic controls
        """
        raise RuntimeError("This method should be implemented in the child class")

    def a_ode(self, nlp) -> MX:
        """
        The symbolic numerical timeseries

        Parameters
        ----------
        nlp
            The NonLinearProgram handler

        Returns
        -------
        The symbolic algebraic variables
        """
        raise RuntimeError("This method should be implemented in the child class")

    def param_ode(self, nlp) -> MX:
        """
        The symbolic parameters

        Parameters
        ----------
        nlp
            The NonLinearProgram handler

        Returns
        -------
        The symbolic parameters
        """
        return nlp.parameters.scaled.cx_start

    def d_ode(self, nlp) -> MX:
        """
        The symbolic algebraic states variables

        Parameters
        ----------
        nlp
            The NonLinearProgram handler

        Returns
        -------
        The symbolic numerical timeseries
        """
        raise RuntimeError("This method should be implemented in the child class")

    def initialize_integrator(
        self, ocp, nlp, dynamics_index: int, node_index: int, is_extra_dynamics: bool = False, **extra_opt
    ) -> Callable:
        """
        Initialize the integrator

        Parameters
        ----------
        ocp
            The Optimal control program handler
        nlp
            The NonLinearProgram handler
        node_index
            The index of the node currently initialized
        is_implicit
            If the dynamics is implicit or not
        dynamics_index
            The current extra dynamics to resolve (that can be referred to nlp.extra_dynamics_func[index])
        is_extra_dynamics
            If the dynamics is an extra dynamics
        extra_opt
            Any extra options to pass to the integrator

        Returns
        -------
        The initialized integrator function
        """

        if dynamics_index > 0 and not is_extra_dynamics:
            raise RuntimeError("dynamics_index should be 0 if is_extra_dynamics is False")

        nlp.states.node_index = node_index
        nlp.states_dot.node_index = node_index
        nlp.controls.node_index = node_index
        nlp.algebraic_states.node_index = node_index

        if nlp.dynamics_func is None:
            dynamics_func = None
        elif is_extra_dynamics:
            dynamics_func = nlp.extra_dynamics_func[dynamics_index]
        else:
            dynamics_func = nlp.dynamics_func

        if nlp.dynamics_defects_func is None:
            dynamics_defects_func = None
        elif is_extra_dynamics:
            dynamics_defects_func = nlp.extra_dynamics_defects_func[dynamics_index]
        else:
            dynamics_defects_func = nlp.dynamics_defects_func

        ode_index = None
        if dynamics_func is not None:
            ode_index = node_index if dynamics_func.size2_out("xdot") > 1 else 0
        ode_opt = {
            "model": nlp.model,
            "cx": nlp.cx,
            "control_type": nlp.control_type,
            "defects_type": self.defects_type,
            "ode_index": ode_index,
            "duplicate_starting_point": self.duplicate_starting_point,
            **extra_opt,
        }

        ode = {
            "t": self.t_ode(nlp),
            "x": self.x_ode(nlp),
            "u": self.p_ode(nlp),
            "a": self.a_ode(nlp),
            "d": self.d_ode(nlp),
            "param": self.param_ode(nlp),
            "ode": dynamics_func,
            "implicit_ode": dynamics_defects_func,
        }

        return nlp.ode_solver.integrator(ode, ode_opt)

    def prepare_dynamic_integrator(self, ocp, nlp):
        """
        Properly set the integration in a nlp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the main program
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        # Primary dynamics
        dynamics = [nlp.ode_solver.initialize_integrator(ocp, nlp, dynamics_index=0, node_index=0)]
        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
            dynamics = dynamics * nlp.ns
        else:
            for node_index in range(1, nlp.ns):
                dynamics.append(nlp.ode_solver.initialize_integrator(ocp, nlp, dynamics_index=0, node_index=node_index))
        nlp.dynamics = dynamics

        # Extra dynamics
        extra_dynamics = []
        for i in range(len(nlp.extra_dynamics_func)):
            extra_dynamics += [
                nlp.ode_solver.initialize_integrator(ocp, nlp, dynamics_index=i, node_index=0, is_extra_dynamics=True)
            ]
            if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
                extra_dynamics = extra_dynamics * nlp.ns
            else:
                for node_index in range(1, nlp.ns):
                    extra_dynamics += [
                        nlp.ode_solver.initialize_integrator(
                            ocp, nlp, dynamics_index=i, node_index=node_index, is_extra_dynamics=True
                        )
                    ]
            nlp.extra_dynamics.append(extra_dynamics)

        # Extra dynamics
        extra_dynamics_defects = []
        for i in range(len(nlp.extra_dynamics_defects_func)):
            extra_dynamics_defects += [
                nlp.ode_solver.initialize_integrator(ocp, nlp, dynamics_index=i, node_index=0, is_extra_dynamics=True)
            ]
            if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
                extra_dynamics_defects = extra_dynamics_defects * nlp.ns
            else:
                for node_index in range(1, nlp.ns):
                    extra_dynamics_defects += [
                        nlp.ode_solver.initialize_integrator(
                            ocp, nlp, dynamics_index=i, node_index=node_index, is_extra_dynamics=True
                        )
                    ]
            nlp.extra_dynamics_defects.append(extra_dynamics_defects)
