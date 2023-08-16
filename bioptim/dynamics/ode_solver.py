from typing import Callable

from casadi import MX, SX, integrator as casadi_integrator, horzcat, Function

from .integrator import RK1, RK2, RK4, RK8, IRK, COLLOCATION, CVODES, TRAPEZOIDAL
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
    integrator(self, ocp, nlp, node_index) -> list
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

    def integrator(self, ocp, nlp, node_index: int, with_noise: bool = False) -> list:
        """
        The interface of the OdeSolver to the corresponding integrator

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the nlp
        node_index
            The index of the node currently evaluated
        with_noise: bool
            If the noise should be added to the ode

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
        nlp.dynamics += nlp.ode_solver.integrator(ocp, nlp, node_index=0, with_noise=False)
        if ocp.assume_phase_dynamics:
            nlp.dynamics = nlp.dynamics * nlp.ns
        else:
            for node_index in range(1, nlp.ns):
                nlp.dynamics += nlp.ode_solver.integrator(ocp, nlp, node_index, with_noise=False)

    @staticmethod
    def prepare_noised_dynamic_integrator(ocp, nlp):
        """
        Properly set the integration of the noised dynamics in an nlp

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the main program
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """
        nlp.noised_dynamics = []
        nlp.noised_dynamics += nlp.ode_solver.integrator(ocp, nlp, node_index=0, with_noise=True)
        if ocp.assume_phase_dynamics:
            nlp.noised_dynamics = nlp.nosed_dynamics * nlp.ns
        else:
            for node_index in range(1, nlp.ns):
                nlp.noised_dynamics += nlp.ode_solver.integrator(ocp, nlp, node_index, with_noise=True)


class RK(OdeSolverBase):
    """
    The base class for Runge-Kutta

    Methods
    -------
    integrator(self, ocp, nlp, node_index, with_noise) -> list
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

    def integrator(self, ocp, nlp, node_index: int, with_noise: bool = False) -> list:
        """
        The interface of the OdeSolver to the corresponding integrator

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the nlp
        node_index
            The index of the node currently integrated
        with_noise: bool
            If the noise should be added to the dynamics

        Returns
        -------
        A list of integrators
        """

        nlp.states.node_index = node_index
        nlp.states_dot.node_index = node_index
        nlp.controls.node_index = node_index
        nlp.stochastic_variables.node_index = node_index

        motor_noise = ocp.cx()
        sensory_noise = ocp.cx()
        if with_noise:
            if nlp.motor_noise is None:
                raise RuntimeError(
                    "You can only call integrator with_noise=True while running a " "StochasticOptimalControlProgram."
                )
            motor_noise = nlp.motor_noise
            sensory_noise = nlp.sensory_noise

        ode_opt = {
            "t0": 0,
            "tf": nlp.dt,
            "model": nlp.model,
            "param": nlp.parameters,
            "motor_noise": motor_noise,
            "sensory_noise": sensory_noise,
            "cx": nlp.cx,
            "idx": 0,
            "control_type": nlp.control_type,
            "number_of_finite_elements": self.steps,
            "defects_type": DefectType.NOT_APPLICABLE,
        }

        ode = {
            "x_unscaled": nlp.states.cx_start,
            "x_scaled": nlp.states.scaled.cx_start,
            "p_unscaled": nlp.controls.cx_start
            if nlp.control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE, ControlType.NONE)
            else horzcat(nlp.controls.cx_start, nlp.controls.cx_end),
            "p_scaled": nlp.controls.scaled.cx_start
            if nlp.control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE, ControlType.NONE)
            else horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end),
            "s_unscaled": nlp.stochastic_variables.cx_start,
            "s_scaled": nlp.stochastic_variables.scaled.cx_start,
            "ode": nlp.dynamics_func,
            "noised_ode": nlp.noised_dynamics_func if with_noise else None,
            "implicit_ode": nlp.implicit_dynamics_func,
            "noised_implicit_ode": nlp.noised_implicit_dynamics_func if with_noise else None,
        }

        if ode["ode"].size2_out("xdot") != 1:
            # If the ode is designed for each node, use the proper node, otherwise use the first one
            # Please note this is unrelated to ocp.assume_phase_dynamics
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

    class TRAPEZOIDAL(OdeSolverBase):
        """
        A trapezoidal ode solver

        Methods
        -------
        integrator(self, ocp, nlp, node_index, with_noise) -> list
            The interface of the OdeSolver to the corresponding integrator
        """

        def __init__(self):
            super(OdeSolver.TRAPEZOIDAL, self).__init__()
            self.rk_integrator = TRAPEZOIDAL
            self.is_direct_shooting = True
            self.defects_type = DefectType.NOT_APPLICABLE

        def integrator(self, ocp, nlp, node_index: int, with_noise: bool = False) -> list:
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp
            node_index
                The index of the node currently integrated
            with_noise: bool
                If the noise should be added to the ode

            Returns
            -------
            A list of integrators
            """

            nlp.states.node_index = node_index
            nlp.states_dot.node_index = node_index
            nlp.controls.node_index = node_index
            nlp.stochastic_variables.node_index = node_index

            if nlp.control_type == ControlType.CONSTANT:
                raise RuntimeError(
                    "TRAPEZOIDAL cannot be used with piece-wise constant controls, please use "
                    "ControlType.CONSTANT_WITH_LAST_NODE or ControlType.LINEAR_CONTINUOUS instead."
                )

            motor_noise = ocp.cx()
            sensory_noise = ocp.cx()
            if with_noise:
                if nlp.motor_noise is None:
                    raise RuntimeError(
                        "You can only call integrator with_noise=True while running a "
                        "StochasticOptimalControlProgram."
                    )
                motor_noise = nlp.motor_noise
                sensory_noise = nlp.sensory_noise

            ode = {
                "x_unscaled": horzcat(nlp.states.cx_start, nlp.states.cx_end),
                "x_scaled": horzcat(nlp.states.scaled.cx_start, nlp.states.scaled.cx_end),
                "p_unscaled": horzcat(nlp.controls.cx_start, nlp.controls.cx_end),
                "p_scaled": horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end),
                "s_unscled": horzcat(nlp.stochastic_variables.cx_start, nlp.stochastic_variables.cx_end),
                "s_scaled": horzcat(nlp.stochastic_variables.scaled.cx_start, nlp.stochastic_variables.scaled.cx_end),
                "ode": nlp.dynamics_func,
                "noised_ode": nlp.noised_dynamics_func if with_noise else None,
                "implicit_ode": nlp.implicit_dynamics_func,
                "noised_implicit_ode": nlp.noised_implicit_dynamics_func if with_noise else None,
            }
            ode_opt = {
                "t0": 0,
                "tf": nlp.dt,
                "model": nlp.model,
                "param": nlp.parameters,
                "motor_noise": motor_noise,
                "sensory_noise": sensory_noise,
                "cx": nlp.cx,
                "idx": 0,
                "control_type": nlp.control_type,
                "defects_type": DefectType.NOT_APPLICABLE,
            }

            if ode["ode"].size2_out("xdot") != 1:
                ode_opt["idx"] = node_index
            return [nlp.ode_solver.rk_integrator(ode, ode_opt)]

        def __str__(self):
            return f"{self.rk_integrator.__name__}"

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

        def integrator(self, ocp, nlp, node_index: int, with_noise: bool = False) -> list:
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp
            node_index
                The index of the node currently integrated
            with_noise: bool
                If the noise should be added to the dynamics

            Returns
            -------
            A list of integrators
            """

            nlp.states.node_index = node_index
            nlp.states_dot.node_index = node_index
            nlp.controls.node_index = node_index
            nlp.stochastic_variables.node_index = node_index

            if ocp.n_threads > 1 and nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("Piece-wise linear continuous controls cannot be used with multiple threads")

            if nlp.model.nb_quaternions > 0:
                raise NotImplementedError(
                    "Quaternions can't be used with IRK yet. If you get this error, please notify the "
                    "developers and ping @EveCharbie"
                )

            motor_noise = ocp.cx()
            sensory_noise = ocp.cx()
            if with_noise:
                if nlp.motor_noise is None:
                    raise RuntimeError(
                        "You can only call integrator with_noise=True while running a "
                        "StochasticOptimalControlProgram."
                    )
                motor_noise = nlp.motor_noise
                sensory_noise = nlp.sensory_noise

            ode = {
                "x_unscaled": [nlp.states.cx_start] + nlp.states.cx_intermediates_list,
                "x_scaled": [nlp.states.scaled.cx_start] + nlp.states.scaled.cx_intermediates_list,
                "p_unscaled": nlp.controls.cx_start,
                "p_scaled": nlp.controls.scaled.cx_start,
                "s_unscaled": nlp.stochastic_variables.cx_start,
                "s_scaled": nlp.stochastic_variables.scaled.cx_start,
                "ode": nlp.dynamics_func,
                "noised_ode": nlp.noised_dynamics_func if with_noise else None,
                "implicit_ode": nlp.implicit_dynamics_func,
                "noised_implicit_ode": nlp.noised_implicit_dynamics_func if with_noise else None,
            }
            ode_opt = {
                "t0": 0,
                "tf": nlp.dt,
                "model": nlp.model,
                "param": nlp.parameters,
                "motor_noise": motor_noise,
                "sensory_noise": sensory_noise,
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

        def integrator(self, ocp, nlp, node_index: int, with_noise: bool = False) -> list:
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp
            node_index
                The index of the node currently integrated

            Returns
            -------
            A list of integrators
            """

            if ocp.cx is SX:
                raise RuntimeError("use_sx=True and OdeSolver.IRK are not yet compatible")

            return super(OdeSolver.IRK, self).integrator(ocp, nlp, node_index, with_noise=with_noise)

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

        def integrator(self, ocp, nlp, node_index: int, with_noise: bool = False) -> list:
            """
            The interface of the OdeSolver to the corresponding integrator

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the nlp
            node_index
                The index of the node currently integrated

            Returns
            -------
            A list of integrators
            """

            nlp.states.node_index = node_index
            nlp.states_dot.node_index = node_index
            nlp.controls.node_index = node_index
            nlp.stochastic_variables.node_index = node_index

            if not isinstance(ocp.cx(), MX):
                raise RuntimeError("use_sx=True and OdeSolver.CVODES are not yet compatible")
            if ocp.parameters.shape != 0:
                raise RuntimeError("CVODES cannot be used while optimizing parameters")
            if nlp.stochastic_variables.cx_start.shape != 0 and nlp.stochastic_variables.cx_start.shape != (0, 0):
                raise RuntimeError("CVODES cannot be used while optimizing stochastic variables")
            if nlp.external_forces:
                raise RuntimeError("CVODES cannot be used with external_forces")
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("CVODES cannot be used with piece-wise linear controls (only RK4)")
            if nlp.stochastic_variables.shape != 0:
                raise RuntimeError("CVODES cannot be used with stochastic variables")
            if with_noise:
                raise RuntimeError("CVODES cannot be used with_noise=True")

            ode = {
                "x": nlp.states.scaled.cx_start,
                "p": nlp.controls.scaled.cx_start,
                "ode": nlp.dynamics_func(
                    nlp.states.scaled.cx_start,
                    nlp.controls.scaled.cx_start,
                    nlp.parameters.cx,
                    nlp.stochastic_variables.scaled.cx_start,
                    nlp.cx(),
                    nlp.cx(),
                ),
            }
            ode_opt = {"t0": 0, "tf": nlp.dt}

            integrator_func = casadi_integrator("integrator", "cvodes", ode, ode_opt)

            return [
                Function(
                    "integrator",
                    [
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                        nlp.parameters.cx,
                        nlp.stochastic_variables.scaled.cx_start,
                    ],
                    self._adapt_integrator_output(
                        integrator_func,
                        nlp.states.scaled.cx_start,
                        nlp.controls.scaled.cx_start,
                    ),
                    ["x0", "p", "params", "s"],
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
            s: MX | SX
                Symbolic variable of stochastic variables

            Returns
            -------
            xf and xall
            """

            xf = integrator_func(x0=x0, p=p)["xf"]
            return xf, horzcat(x0, xf)

        def __str__(self):
            return self.rk_integrator.__name__
