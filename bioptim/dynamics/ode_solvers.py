from typing import Callable, TYPE_CHECKING

from casadi import MX, SX, integrator as casadi_integrator, horzcat, Function, vertcat

from . import integrator
from .ode_solver_base import OdeSolverBase
from .rk_base import RK
from ..misc.enums import ControlType, DefectType
from ..misc.parameters_types import Bool, Int, Str, CX

if TYPE_CHECKING:
    from ..models.protocols.biomodel import BioModel
    from ..optimization.parameters import ParameterList


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

    class VARIATIONAL(OdeSolverBase):
        """
        The variational integrator (discrete mechanics). Instead of integrating an ode over each shooting interval,
        the discrete Euler-Lagrange equations link the positions of three consecutive nodes. The states of the problem
        are therefore only the positions (and Lagrange multipliers) at the nodes, and the initial and final velocities
        are parameters (qdot_start and qdot_end).
        """

        @property
        def is_direct_collocation(self) -> Bool:
            return False

        @property
        def is_direct_shooting(self) -> Bool:
            return True

        @property
        def n_required_cx(self) -> Int:
            return 1

        @property
        def defects_type(self) -> DefectType:
            return DefectType.NOT_APPLICABLE

        @property
        def integrates_ode(self) -> Bool:
            return False

        def x_ode(self, nlp):
            return nlp.states.scaled.cx_start

        def p_ode(self, nlp):
            return horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end)

        def a_ode(self, nlp):
            return nlp.algebraic_states.scaled.cx_start

        def d_ode(self, nlp):
            return nlp.numerical_timeseries.cx_start

        def __str__(self) -> Str:
            return f"{self.integrator.__name__} (variational integrator)"

        def declare_parameters(self, parameters: "ParameterList", model: "BioModel") -> None:
            """
            Declare the initial and final velocities, which are not states of the variational integrator.

            Parameters
            ----------
            parameters: ParameterList
                The parameters of the ocp
            model: BioModel
                The model of the phase

            Raises
            ------
            KeyError
                If 'qdot_start' or 'qdot_end' is already declared
            """
            names = ("qdot_start", "qdot_end")
            if any(name in parameters.keys() for name in names):
                raise KeyError(
                    "'qdot_start' and 'qdot_end' cannot be declared in parameters as they are reserved words of "
                    "OdeSolver.VARIATIONAL. To define the initial and final velocities, please use `qdot_init` "
                    "and `qdot_bounds` instead."
                )
            for name in names:
                parameters.add(name, function=None, size=model.nb_q)

        def prepare_dynamic_integrator(self, ocp, nlp):
            self._declare_discrete_functions(nlp)
            super().prepare_dynamic_integrator(ocp, nlp)

        @staticmethod
        def _declare_discrete_functions(nlp) -> None:
            """
            Build the casadi functions of the discrete Euler-Lagrange equations, that depend on the nodes i-1, i and
            i+1 and can therefore not be expressed as a dynamics of a single node. They are stored in
            nlp.variational_three_nodes_func (main nodes), nlp.variational_first_node_func (initial boundary condition)
            and nlp.variational_last_node_func (final boundary condition).

            Parameters
            ----------
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            """
            model = nlp.model
            n_q = model.nb_q

            dt = nlp.cx.sym("time_step")
            q_prev = nlp.cx.sym("q_prev", n_q, 1)
            q_cur = nlp.cx.sym("q_cur", n_q, 1)
            q_next = nlp.cx.sym("q_next", n_q, 1)
            control_prev = nlp.cx.sym("control_prev", n_q, 1)
            control_cur = nlp.cx.sym("control_cur", n_q, 1)
            control_next = nlp.cx.sym("control_next", n_q, 1)
            q0 = nlp.cx.sym("q0", n_q, 1)
            qdot0 = nlp.cx.sym("qdot_start", n_q, 1)
            q1 = nlp.cx.sym("q1", n_q, 1)
            control0 = nlp.cx.sym("control0", n_q, 1)
            control1 = nlp.cx.sym("control1", n_q, 1)
            q_ultimate = nlp.cx.sym("q_ultimate", n_q, 1)
            qdot_ultimate = nlp.cx.sym("qdot_ultimate", n_q, 1)
            q_penultimate = nlp.cx.sym("q_penultimate", n_q, 1)
            controlN_minus_1 = nlp.cx.sym("controlN_minus_1", n_q, 1)
            controlN = nlp.cx.sym("controlN", n_q, 1)

            three_nodes_input = [dt, q_prev, q_cur, q_next, control_prev, control_cur, control_next]
            first_node_input = [dt, q0, qdot0, q1, control0, control1]
            last_node_input = [dt, q_penultimate, q_ultimate, qdot_ultimate, controlN_minus_1, controlN]

            if model.has_holonomic_constraints:
                lambdas = nlp.cx.sym("lambda", model.nb_holonomic_constraints, 1)
                three_nodes_input.append(lambdas)
                first_node_input.append(lambdas)
                last_node_input.append(lambdas)
            else:
                lambdas = None

            # TODO: see how to restore the possibility to not expand the functions
            nlp.variational_three_nodes_func = Function(
                "ThreeNodesIntegration",
                three_nodes_input,
                [
                    model.discrete_euler_lagrange_equations(
                        dt, q_prev, q_cur, q_next, control_prev, control_cur, control_next, lambdas
                    )
                ],
            ).expand()

            nlp.variational_first_node_func = Function(
                "TwoFirstNodesIntegration",
                first_node_input,
                [model.compute_initial_states(dt, q0, qdot0, q1, control0, control1, lambdas)],
            ).expand()

            nlp.variational_last_node_func = Function(
                "TwoLastNodesIntegration",
                last_node_input,
                [
                    model.compute_final_states(
                        dt, q_penultimate, q_ultimate, qdot_ultimate, controlN_minus_1, controlN, lambdas
                    )
                ],
            ).expand()

        @property
        def integrator(self):
            return integrator.VARIATIONAL

    class TRAPEZOIDAL(OdeSolverBase):
        """
        A trapezoidal ode solver
        """

        @property
        def integrator(self):
            return integrator.TRAPEZOIDAL

        @property
        def is_direct_collocation(self) -> Bool:
            return False

        @property
        def is_direct_shooting(self) -> Bool:
            return True

        @property
        def defects_type(self) -> DefectType:
            return DefectType.NOT_APPLICABLE

        @property
        def n_required_cx(self) -> Int:
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

        def __str__(self) -> Str:
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
            The type of defect to use
        duplicate_starting_point: bool
            Whether an additional collocation point should be added at the shooting node (this is typically used in SOCPs)
        """

        def __init__(
            self,
            polynomial_degree: Int = 4,
            method: Str = "legendre",
            defects_type: DefectType = DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS,
            **kwargs,
        ):
            """
            Parameters
            ----------
            polynomial_degree: int
                The degree of the implicit RK
            """

            if not isinstance(defects_type, DefectType):
                raise TypeError("defects_type should be a DefectType")

            super(OdeSolver.COLLOCATION, self).__init__(**kwargs)
            self.polynomial_degree = polynomial_degree
            self.method = method
            self._defects_type = defects_type

        @property
        def integrator(self):
            return integrator.COLLOCATION

        @property
        def is_direct_shooting(self) -> Bool:
            return False

        @property
        def is_direct_collocation(self) -> Bool:
            return True

        @property
        def n_required_cx(self) -> Int:
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
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                return horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end)
            elif nlp.control_type == ControlType.NONE:
                return nlp.cx()
            else:
                raise NotImplementedError(f"The control_type {nlp.control_type} is not implemented.")

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

        def __str__(self) -> Str:
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
        def is_direct_collocation(self) -> Bool:
            return False

        @property
        def is_direct_shooting(self) -> Bool:
            return True

    class CVODES(OdeSolverBase):
        """
        An interface to CVODES
        """

        @property
        def integrator(self):
            return integrator.CVODES

        @property
        def is_direct_collocation(self) -> Bool:
            return False

        @property
        def is_direct_shooting(self) -> Bool:
            return True

        @property
        def n_required_cx(self) -> Int:
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

        def initialize_integrator(self, ocp, nlp, dynamics_index: Int, node_index: Int, **extra_opt):
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
        def _adapt_integrator_output(integrator_func: Callable, x0: CX, u: CX):
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

        def __str__(self) -> Str:
            return self.integrator.__name__
