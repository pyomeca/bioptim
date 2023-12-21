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

    def __init__(self, allow_free_variables: bool = False, duplicate_starting_point: bool = False):
        """
        Parameters
        ----------
        allow_free_variables: bool
            If the free variables are allowed in the integrator's casadi function
        duplicate_starting_point: bool
            If the starting point should be duplicated in the integrator's casadi function
        """
        self.allow_free_variables = allow_free_variables
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
        The symbolic algebraic states variables

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
        return nlp.parameters.cx

    def initialize_integrator(
        self, ocp, nlp, dynamics_index: int, node_index: int, allow_free_variables: bool = False, **extra_opt
    ) -> Callable:
        """
        Initialize the integrator

        Parameters
        ----------
        ocp
            The Optimal control program handler
        nlp
            The NonLinearProgram handler
        dynamics_index
            The current dynamics to resolve (that can be referred to nlp.dynamics_func[index])
        node_index
            The index of the node currently initialized
        allow_free_variables
            If the free variables are allowed in the integrator's casadi function
        extra_opt
            Any extra options to pass to the integrator

        Returns
        -------
        The initialized integrator function
        """

        nlp.states.node_index = node_index
        nlp.states_dot.node_index = node_index
        nlp.controls.node_index = node_index
        nlp.algebraic_states.node_index = node_index
        ode_opt = {
            "model": nlp.model,
            "cx": nlp.cx,
            "control_type": nlp.control_type,
            "defects_type": self.defects_type,
            "allow_free_variables": allow_free_variables,
            "param_scaling": vertcat(*[nlp.parameters[key].scaling.scaling for key in nlp.parameters.keys()]),
            "ode_index": node_index if nlp.dynamics_func[dynamics_index].size2_out("xdot") > 1 else 0,
            "duplicate_starting_point": self.duplicate_starting_point,
            **extra_opt,
        }

        ode = {
            "t": self.t_ode(nlp),
            "x": self.x_ode(nlp),
            "p": self.p_ode(nlp),
            "a": self.a_ode(nlp),
            "param": self.param_ode(nlp),
            "ode": nlp.dynamics_func[dynamics_index],
            # TODO this actually checks "not nlp.implicit_dynamics_func" (or that nlp.implicit_dynamics_func == [])
            "implicit_ode": nlp.implicit_dynamics_func[dynamics_index]
            if len(nlp.implicit_dynamics_func) > 0
            else nlp.implicit_dynamics_func,
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
        dynamics = [
            nlp.ode_solver.initialize_integrator(
                ocp, nlp, dynamics_index=0, node_index=0, allow_free_variables=self.allow_free_variables
            )
        ]
        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
            dynamics = dynamics * nlp.ns
        else:
            for node_index in range(1, nlp.ns):
                dynamics.append(
                    nlp.ode_solver.initialize_integrator(
                        ocp,
                        nlp,
                        dynamics_index=0,
                        node_index=node_index,
                        allow_free_variables=self.allow_free_variables,
                    )
                )
        nlp.dynamics = dynamics

        # Extra dynamics
        extra_dynamics = []
        for i in range(1, len(nlp.dynamics_func)):
            extra_dynamics += [
                nlp.ode_solver.initialize_integrator(
                    ocp, nlp, dynamics_index=i, node_index=0, allow_free_variables=True
                )
            ]
            if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
                extra_dynamics = extra_dynamics * nlp.ns
            else:
                for node_index in range(1, nlp.ns):
                    extra_dynamics += [
                        nlp.ode_solver.initialize_integrator(
                            ocp, nlp, dynamics_index=i, node_index=0, allow_free_variables=True
                        )
                    ]
            # TODO include this in nlp.dynamics so the index of nlp.dynamics_func and nlp.dynamics match
            nlp.extra_dynamics.append(extra_dynamics)


class RK(OdeSolverBase):
    """
    The base class for Runge-Kutta
    """

    def __init__(self, n_integration_steps: int = 5, **kwargs):
        """
        Parameters
        ----------
        n_integration_steps: int
            The number of steps for the integration
        """

        super(RK, self).__init__(**kwargs)
        self.n_integration_steps = n_integration_steps

    @property
    def is_direct_collocation(self) -> bool:
        return False

    @property
    def is_direct_shooting(self) -> bool:
        return True

    @property
    def n_required_cx(self) -> int:
        return 1

    @property
    def defects_type(self) -> DefectType:
        return DefectType.NOT_APPLICABLE

    def initialize_integrator(self, *args, **kwargs):
        return super(RK, self).initialize_integrator(
            *args, **kwargs, number_of_finite_elements=self.n_integration_steps
        )

    def x_ode(self, nlp):
        return nlp.states.scaled.cx_start

    def p_ode(self, nlp):
        if nlp.control_type in (
            ControlType.CONSTANT,
            ControlType.CONSTANT_WITH_LAST_NODE,
        ):
            return nlp.controls.scaled.cx_start
        else:
            return horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end)

    def a_ode(self, nlp):
        return nlp.algebraic_states.scaled.cx_start

    def __str__(self):
        ode_solver_string = f"{self.integrator.__name__} {self.n_integration_steps} step"
        if self.n_integration_steps > 1:
            ode_solver_string += "s"

        return ode_solver_string


class OdeSolver:
    """
    The public interface to the different OdeSolvers
    """

    class RK1(RK):
        """
        A Runge-Kutta 1 solver (Forward Euler Method)
        """

        @property
        def integrator(self):
            return integrator.RK1

    class RK2(RK):
        """
        A Runge-Kutta 2 solver (Midpoint Method)
        """

        @property
        def integrator(self):
            return integrator.RK2

    class RK4(RK):
        """
        A Runge-Kutta 4 solver
        """

        @property
        def integrator(self):
            return integrator.RK4

    class RK8(RK):
        """
        A Runge-Kutta 8 solver
        """

        @property
        def integrator(self):
            return integrator.RK8

    class TRAPEZOIDAL(OdeSolverBase):
        """
        A trapezoidal ode solver
        """

        @property
        def integrator(self):
            return integrator.TRAPEZOIDAL

        @property
        def is_direct_collocation(self) -> bool:
            return False

        @property
        def is_direct_shooting(self) -> bool:
            return True

        @property
        def defects_type(self) -> DefectType:
            return DefectType.NOT_APPLICABLE

        @property
        def defect_type(self) -> DefectType:
            return DefectType.NOT_APPLICABLE

        @property
        def n_required_cx(self) -> int:
            return 1

        def x_ode(self, nlp):
            return horzcat(nlp.states.scaled.cx_start, nlp.states.scaled.cx_end)

        def p_ode(self, nlp):
            return horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end)

        def a_ode(self, nlp):
            return horzcat(nlp.algebraic_states.scaled.cx_start, nlp.algebraic_states.scaled.cx_end)

        def initialize_integrator(self, ocp, nlp, **kwargs):
            if nlp.control_type == ControlType.CONSTANT:
                raise RuntimeError(
                    "TRAPEZOIDAL cannot be used with piece-wise constant controls, please use "
                    "ControlType.CONSTANT_WITH_LAST_NODE or ControlType.LINEAR_CONTINUOUS instead."
                )
            return super(OdeSolver.TRAPEZOIDAL, self).initialize_integrator(ocp, nlp, **kwargs)

        def __str__(self):
            return f"{self.integrator.__name__}"

    class COLLOCATION(OdeSolverBase):
        """
        An implicit Runge-Kutta solver

        Attributes
        ----------
        polynomial_degree: int
            The degree of the implicit RK
        method : str
            The method of interpolation ("legendre" or "radau")
        defects_type: DefectType
            The type of defect to use (DefectType.EXPLICIT or DefectType.IMPLICIT)
        duplicate_starting_point: bool
            Whether an additional collocation point should be added at the shooting node (this is typically used in SOCPs)
        """

        def __init__(
            self,
            polynomial_degree: int = 4,
            method: str = "legendre",
            defects_type: DefectType = DefectType.EXPLICIT,
            **kwargs,
        ):
            """
            Parameters
            ----------
            polynomial_degree: int
                The degree of the implicit RK
            """

            super(OdeSolver.COLLOCATION, self).__init__(**kwargs)
            self.polynomial_degree = polynomial_degree
            self.method = method
            self._defects_type = defects_type

        @property
        def integrator(self):
            return integrator.COLLOCATION

        @property
        def is_direct_shooting(self) -> bool:
            return False

        @property
        def is_direct_collocation(self) -> bool:
            return True

        @property
        def n_required_cx(self) -> int:
            return self.polynomial_degree + (1 if self.duplicate_starting_point else 0)

        @property
        def defects_type(self) -> DefectType:
            return self._defects_type

        def x_ode(self, nlp):
            out = [nlp.states.scaled.cx_start]
            if not self.duplicate_starting_point:
                out += [nlp.states.scaled.cx_start]
            out += nlp.states.scaled.cx_intermediates_list
            return out

        def p_ode(self, nlp):
            return nlp.controls.scaled.cx_start

        def a_ode(self, nlp):
            return nlp.algebraic_states.scaled.cx_start

        def initialize_integrator(self, ocp, nlp, **kwargs):
            if ocp.n_threads > 1 and nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("Piece-wise linear continuous controls cannot be used with multiple threads")

            if nlp.model.nb_quaternions > 0:
                raise NotImplementedError(
                    "Quaternions can't be used with IRK yet. If you get this error, please notify the "
                    "developers and ping @EveCharbie"
                )

            return super(OdeSolver.COLLOCATION, self).initialize_integrator(
                ocp, nlp, **kwargs, method=self.method, irk_polynomial_interpolation_degree=self.polynomial_degree
            )

        def __str__(self):
            return f"{self.integrator.__name__} {self.method} {self.polynomial_degree}"

    class IRK(COLLOCATION):
        """
        An implicit Runge-Kutta solver
        """

        def initialize_integrator(self, ocp, nlp, **kwargs):
            if ocp.cx is SX:
                raise NotImplementedError("use_sx=True and OdeSolver.IRK are not yet compatible")

            return super(OdeSolver.IRK, self).initialize_integrator(ocp, nlp, **kwargs)

        @property
        def integrator(self):
            return integrator.IRK

        @property
        def is_direct_collocation(self) -> bool:
            return False

        @property
        def is_direct_shooting(self) -> bool:
            return True

    class CVODES(OdeSolverBase):
        """
        An interface to CVODES
        """

        @property
        def integrator(self):
            return integrator.CVODES

        @property
        def is_direct_collocation(self) -> bool:
            return False

        @property
        def is_direct_shooting(self) -> bool:
            return True

        @property
        def n_required_cx(self) -> int:
            return 1

        @property
        def defects_type(self) -> DefectType:
            return DefectType.NOT_APPLICABLE

        def x_ode(self, nlp):
            return nlp.states.scaled.cx

        def p_ode(self, nlp):
            return nlp.controls.scaled.cx

        def a_ode(self, nlp):
            return nlp.algebraic_states.scaled.cx

        def initialize_integrator(
            self, ocp, nlp, dynamics_index: int, node_index: int, allow_free_variables: bool = False, **extra_opt
        ):
            raise NotImplementedError("CVODES is not yet implemented")

            if extra_opt:
                raise RuntimeError("CVODES does not accept extra options")

            if not isinstance(ocp.cx(), MX):
                raise RuntimeError("use_sx=True and OdeSolver.CVODES are not yet compatible")
            if ocp.parameters.shape != 0:
                raise RuntimeError(
                    "CVODES cannot be used while optimizing parameters"
                )  # todo: should accept parameters now
            if nlp.algebraic_states.cx_start.shape != 0 and nlp.algebraic_states.cx_start.shape != (0, 0):
                raise RuntimeError("CVODES cannot be used while optimizing algebraic_states variables")
            if nlp.external_forces:
                raise RuntimeError("CVODES cannot be used with external_forces")
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("CVODES cannot be used with piece-wise linear controls (only RK4)")
            if nlp.algebraic_states.shape != 0:
                raise RuntimeError("CVODES cannot be used with algebraic_states variables")

            t = [self.t_ode(nlp)[0], self.t_ode(nlp)[1] - self.t_ode(nlp)[0]]
            ode = {
                "x": nlp.states.scaled.cx_start,
                "u": nlp.controls.scaled.cx_start,  # todo: add p=parameters
                "ode": nlp.dynamics_func[dynamics_index](
                    vertcat(*t), self.x_ode(nlp), self.p_ode(nlp), self.param_ode(nlp), self.a_ode(nlp)
                ),
            }

            ode_opt = {"t0": t[0], "tf": t[1]}
            integrator_func = casadi_integrator("integrator", "cvodes", ode, ode_opt)

            return [
                Function(
                    "integrator",
                    [vertcat(*t), self.x_ode(nlp), self.p_ode(nlp), self.param_ode(nlp), self.a_ode(nlp)],
                    self._adapt_integrator_output(
                        integrator_func,
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                    ),
                    ["t_span", "x0", "u", "p", "a"],
                    ["xf", "xall"],
                    {"allow_free": allow_free_variables},
                )
            ]

        @staticmethod
        def _adapt_integrator_output(integrator_func: Callable, x0: MX | SX, u: MX | SX):
            """
            Interface to make xf and xall as outputs

            Parameters
            ----------
            integrator_func: Callable
                Handler on a CasADi function
            x0: MX | SX
                Symbolic variable of states
            u: MX | SX
                Symbolic variable of controls

            Returns
            -------
            xf and xall
            """

            xf = integrator_func(x0=x0, u=u)["xf"]
            return xf, horzcat(x0, xf)

        def __str__(self):
            return self.integrator.__name__
