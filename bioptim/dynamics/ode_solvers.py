from typing import Callable

from casadi import MX, SX, integrator as casadi_integrator, horzcat, Function, vertcat

from . import integrator
from .ode_solver_base import OdeSolverBase
from .rk_base import RK
from ..misc.enums import ControlType, DefectType, PhaseDynamics


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

        def d_ode(self, nlp):
            return horzcat(nlp.numerical_timeseries.cx_start, nlp.numerical_timeseries.cx_end)

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
        _defects_type: DefectType
            The type of defects to use to constrain the slope of the polynomial at the COLLOCATION nodes
        duplicate_starting_point: bool
            Whether an additional collocation point should be added at the shooting node (this is typically used in SOCPs)
        """

        def __init__(
            self,
            polynomial_degree: int = 4,
            method: str = "legendre",
            defects_type: DefectType = DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS,
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
            if nlp.control_type in (
                ControlType.CONSTANT,
                ControlType.CONSTANT_WITH_LAST_NODE,
            ):
                return nlp.controls.scaled.cx_start
            else:
                return horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end)

        def a_ode(self, nlp):
            out = [nlp.algebraic_states.scaled.cx_start]
            if not self.duplicate_starting_point:
                out += [nlp.algebraic_states.scaled.cx_start]
            out += nlp.algebraic_states.scaled.cx_intermediates_list
            return out

        def d_ode(self, nlp):
            return nlp.numerical_timeseries.cx_start

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

        def initialize_integrator(self, ocp, nlp, dynamics_index: int, node_index: int, **extra_opt):
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
            if nlp.numerical_timeseries:
                raise RuntimeError("CVODES cannot be used with external_forces or other numerical_timeseries")
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("CVODES cannot be used with piece-wise linear controls (only RK4)")
            if nlp.algebraic_states.shape != 0:
                raise RuntimeError("CVODES cannot be used with algebraic_states variables")

            t = [self.t_ode(nlp)[0], self.t_ode(nlp)[1] - self.t_ode(nlp)[0]]
            dynamics_func = nlp.dynamics_func if not is_extra_dynamics else nlp.extra_dynamics_func[dynamics_index]
            dynamics_defects_func = (
                nlp.dynamics_defects_func
                if not is_extra_dynamics
                else nlp.extra_dynamics_defects_func[dynamics_index]
            )
            ode = {
                "x": nlp.states.scaled.cx_start,
                "u": nlp.controls.scaled.cx_start,  # todo: add p=parameters
                "ode": dynamics_func(
                    vertcat(*t),
                    self.x_ode(nlp),
                    self.p_ode(nlp),
                    self.param_ode(nlp),
                    self.a_ode(nlp),
                    self.d_ode(nlp),
                ),
            }

            ode_opt = {"t0": t[0], "tf": t[1]}
            integrator_func = casadi_integrator("integrator", "cvodes", ode, ode_opt)

            return [
                Function(
                    "integrator",
                    [
                        vertcat(*t),
                        self.x_ode(nlp),
                        self.p_ode(nlp),
                        self.param_ode(nlp),
                        self.a_ode(nlp),
                        self.d_ode(nlp),
                    ],
                    self._adapt_integrator_output(
                        integrator_func,
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                        nlp.numerical_timeseries.cx_start,
                    ),
                    ["t_span", "x0", "u", "p", "a", "d"],
                    ["xf", "xall"],
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
