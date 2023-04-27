from typing import Callable

from casadi import MX, SX, integrator as casadi_integrator, horzcat, Function

from .integrator import RK1, RK2, RK4, RK8, IRK, COLLOCATION, CVODES
from ..misc.enums import ControlType, DefectType


class OdeSolverBase:
    """
    The base class for the ODE solvers

    Attributes
    ----------
    steps: int
        The number of integration steps
    steps_scipy: int
        Number of steps while integrating with scipy
    rk_integrator: RK4 | RK8 | IRK
        The corresponding integrator class
    is_direct_collocation: bool
        indicating if the ode solver is direct collocation method
    is_direct_shooting: bool
        indicating if the ode solver is direct shooting method
    Methods
    -------
    integrator(self, ocp, nlp) -> list
        The interface of the OdeSolver to the corresponding integrator
    prepare_dynamic_integrator(ocp, nlp)
        Properly set the integration in an nlp
    """

    def __init__(self):
        self.steps = 1
        self.steps_scipy = 5
        self.rk_integrator = None
        self.is_direct_collocation = False
        self.is_direct_shooting = False

    def integrator(self, ocp, nlp, node_index=None) -> list:
        """
        The interface of the OdeSolver to the corresponding integrator

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the nlp

        Returns
        -------
        A list of integrators
        """

        raise RuntimeError("OdeSolveBase is abstract, please select a valid OdeSolver")

    @staticmethod
    def prepare_dynamic_integrator(ocp, nlp):
        """
        Properly set the integration in an nlp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the main program
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """
        nlp.dynamics = []
        nlp.dynamics += nlp.ode_solver.integrator(ocp, nlp, 0)
        if ocp.assume_phase_dynamics:
            nlp.dynamics = nlp.dynamics * nlp.ns
        else:
            for node_index in range(1, nlp.ns):
                nlp.dynamics += nlp.ode_solver.integrator(ocp, nlp, node_index)


class RK(OdeSolverBase):
    """
    The base class for Runge-Kutta

    Methods
    -------
    integrator(self, ocp, nlp) -> list
        The interface of the OdeSolver to the corresponding integrator
    """

    def __init__(self, n_integration_steps):
        """
        Parameters
        ----------
        n_integration_steps: int
            The number of steps for the integration
        """

        super(RK, self).__init__()
        self.steps = n_integration_steps
        self.is_direct_shooting = True
        self.defects_type = DefectType.NOT_APPLICABLE

    def integrator(self, ocp, nlp, node_index=None) -> list:
        """
        The interface of the OdeSolver to the corresponding integrator

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the nlp

        Returns
        -------
        A list of integrators
        """

        ode_opt = {
            "t0": 0,
            "tf": nlp.dt,
            "model": nlp.model,
            "param": nlp.parameters,
            "cx": nlp.cx,
            "idx": 0,
            "control_type": nlp.control_type,
            "number_of_finite_elements": self.steps,
            "defects_type": DefectType.NOT_APPLICABLE,
        }

        ode = {
            "x_unscaled": nlp.states[0].cx_start,
            "x_scaled": nlp.states[0]["scaled"].cx_start,
            "p_unscaled": nlp.controls[0].cx_start
            if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE)
            else horzcat(nlp.controls[0].cx_start, nlp.controls[0].cx_end),  # TODO : [0] to [node_index]
            "p_scaled": nlp.controls[0]["scaled"].cx_start
            if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE)
            else horzcat(
                nlp.controls[0]["scaled"].cx_start, nlp.controls[0]["scaled"].cx_end
            ),  # TODO: [0] to [node_index]
            "ode": nlp.dynamics_func,
            "implicit_ode": nlp.implicit_dynamics_func,
        }

        if ode["ode"].size2_out("xdot") != 1:
            ode_opt["idx"] = node_index
        return [nlp.ode_solver.rk_integrator(ode, ode_opt)]

    def __str__(self):
        ode_solver_string = f"{self.rk_integrator.__name__} {self.steps} step"
        if self.steps > 1:
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

        def __init__(self, n_integration_steps: int = 5):
            """
            Parameters
            ----------
            n_integration_steps: int
                The number of steps for the integration
            """

            super(OdeSolver.RK1, self).__init__(n_integration_steps)
            self.rk_integrator = RK1

    class RK2(RK):
        """
        A Runge-Kutta 2 solver (Midpoint Method)
        """

        def __init__(self, n_integration_steps: int = 5):
            """
            Parameters
            ----------
            n_integration_steps: int
                The number of steps for the integration
            """

            super(OdeSolver.RK2, self).__init__(n_integration_steps)
            self.rk_integrator = RK2

    class RK4(RK):
        """
        A Runge-Kutta 4 solver
        """

        def __init__(self, n_integration_steps: int = 5):
            """
            Parameters
            ----------
            n_integration_steps: int
                The number of steps for the integration
            """

            super(OdeSolver.RK4, self).__init__(n_integration_steps)
            self.rk_integrator = RK4

    class RK8(RK):
        """
        A Runge-Kutta 8 solver
        """

        def __init__(self, n_integration_steps: int = 5):
            """
            Parameters
            ----------
            n_integration_steps: int
                The number of steps for the integration
            """

            super(OdeSolver.RK8, self).__init__(n_integration_steps)
            self.rk_integrator = RK8

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

        Methods
        -------
        integrator(self, ocp, nlp) -> list
            The interface of the OdeSolver to the corresponding integrator
        """

        def __init__(
            self, polynomial_degree: int = 4, method: str = "legendre", defects_type: DefectType = DefectType.EXPLICIT
        ):
            """
            Parameters
            ----------
            polynomial_degree: int
                The degree of the implicit RK
            """

            super(OdeSolver.COLLOCATION, self).__init__()
            self.polynomial_degree = polynomial_degree
            self.rk_integrator = COLLOCATION
            self.method = method
            self.defects_type = defects_type
            self.is_direct_collocation = True
            self.steps = self.polynomial_degree

        def integrator(self, ocp, nlp, node_index=None) -> list:
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp
            Returns
            -------
            A list of integrators
            """

            if ocp.n_threads > 1 and nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("Piece-wise linear continuous controls cannot be used with multiple threads")

            if nlp.model.nb_quaternions > 0:
                raise NotImplementedError(
                    "Quaternions can't be used with IRK yet. If you get this error, please notify the "
                    "developers and ping @EveCharbie"
                )

            ode = {
                "x_unscaled": [nlp.states[0].cx_start]
                + nlp.states[0].cx_intermediates_list,  # TODO: [0] to [node_index]
                "x_scaled": [nlp.states[0]["scaled"].cx_start]
                + nlp.states[0]["scaled"].cx_intermediates_list,  # TODO: [0] to [node_index]
                "p_unscaled": nlp.controls[0].cx_start,  # TODO: [0] to [node_index]
                "p_scaled": nlp.controls[0]["scaled"].cx_start,  # TODO: [0] to [node_index]
                "ode": nlp.dynamics_func,
                "implicit_ode": nlp.implicit_dynamics_func,
            }
            ode_opt = {
                "t0": 0,
                "tf": nlp.dt,
                "model": nlp.model,
                "param": nlp.parameters,
                "cx": nlp.cx,
                "idx": 0,
                "control_type": nlp.control_type,
                "irk_polynomial_interpolation_degree": self.polynomial_degree,
                "method": self.method,
                "defects_type": self.defects_type,
            }

            if ode["ode"].size2_out("xdot") != 1:
                ode_opt["idx"] = node_index
            return [nlp.ode_solver.rk_integrator(ode, ode_opt)]

        def __str__(self):
            return f"{self.rk_integrator.__name__} {self.method} {self.polynomial_degree}"

    class IRK(COLLOCATION):
        """
        An implicit Runge-Kutta solver

        Attributes
        ----------
        polynomial_degree: int
            The degree of the implicit RK
        method: str
            The method of interpolation ("legendre" or "radau")
        defects_type: DefectType
            The type of defect to use (DefectType.EXPLICIT or DefectType.IMPLICIT)

        Methods
        -------
        integrator(self, ocp, nlp) -> list
            The interface of the OdeSolver to the corresponding integrator
        """

        def __init__(
            self, polynomial_degree: int = 4, method: str = "legendre", defects_type: DefectType = DefectType.EXPLICIT
        ):
            """
            Parameters
            ----------
            polynomial_degree: int
                The degree of the implicit RK
            """

            super(OdeSolver.IRK, self).__init__(
                polynomial_degree=polynomial_degree, method=method, defects_type=defects_type
            )
            self.rk_integrator = IRK
            self.is_direct_collocation = False
            self.is_direct_shooting = True
            self.steps = 1

        def integrator(self, ocp, nlp, node_index=None) -> list:
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp

            Returns
            -------
            A list of integrators
            """

            if ocp.cx is SX:
                raise RuntimeError("use_sx=True and OdeSolver.IRK are not yet compatible")

            return super(OdeSolver.IRK, self).integrator(ocp, nlp, node_index)

    class CVODES(OdeSolverBase):
        """
        An interface to CVODES
        """

        def __init__(self):
            super(OdeSolver.CVODES, self).__init__()
            self.rk_integrator = CVODES
            self.is_direct_collocation = False
            self.is_direct_shooting = True
            self.steps = 1
            self.defects_type = DefectType.NOT_APPLICABLE

        def integrator(self, ocp, nlp, node_index=None) -> list:  # TODO: add node_index
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp
            Returns
            -------
            A list of integrators
            """
            if not isinstance(ocp.cx(), MX):
                raise RuntimeError("use_sx=True and OdeSolver.CVODES are not yet compatible")
            if ocp.v.parameters_in_list.shape != 0:
                raise RuntimeError("CVODES cannot be used while optimizing parameters")
            if nlp.external_forces:
                raise RuntimeError("CVODES cannot be used with external_forces")
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("CVODES cannot be used with piece-wise linear controls (only RK4)")

            ode = {
                "x": nlp.states[0]["scaled"].cx_start,  # TODO: [0] to [node_index]
                "p": nlp.controls[0]["scaled"].cx_start,  # TODO: [0] to [node_index]
                "ode": nlp.dynamics_func(
                    nlp.states[0]["scaled"].cx_start, nlp.controls[0]["scaled"].cx_start, nlp.parameters.cx_start
                ),  # TODO: [0] to [node_index]
            }
            ode_opt = {"t0": 0, "tf": nlp.dt}

            integrator_func = casadi_integrator("integrator", "cvodes", ode, ode_opt)

            return [
                Function(
                    "integrator",
                    [
                        nlp.states[0]["scaled"].cx_start,
                        nlp.controls[0]["scaled"].cx_start,
                        nlp.parameters.cx_start,
                    ],  # TODO: [0] to [node_index]
                    self._adapt_integrator_output(
                        integrator_func, nlp.states[0]["scaled"].cx_start, nlp.controls[0]["scaled"].cx_start
                    ),  # TODO: [0] to [node_index]
                    ["x0", "p", "params"],
                    ["xf", "xall"],
                )
            ]

        @staticmethod
        def _adapt_integrator_output(integrator_func: Callable, x0: MX | SX, p: MX | SX):
            """
            Interface to make xf and xall as outputs

            Parameters
            ----------
            integrator_func: Callable
                Handler on a CasADi function
            x0: MX | SX
                Symbolic variable of states
            p: MX | SX
                Symbolic variable of controls

            Returns
            -------
            xf and xall
            """

            xf = integrator_func(x0=x0, p=p)["xf"]
            return xf, horzcat(x0, xf)

        def __str__(self):
            return self.rk_integrator.__name__
