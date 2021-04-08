import casadi

from .integrator import RK4, RK8, IRK
from ..misc.enums import ControlType


class OdeSolverBase:
    """
    The base class for the ODE solvers

    Attributes
    ----------
    steps: int
        The number of integration steps
    rk_integrator: Union[RK4, RK8, IRK]
        The corresponding integrator class

    Methods
    -------
    integrator(self, ocp, nlp) -> list
        The interface of the OdeSolver to the corresponding integrator
    @staticmethod
    prepare_dynamic_integrator(ocp, nlp)
        Properly set the integration in an nlp
    """

    def __init__(self):
        self.steps = 1
        self.rk_integrator = None

    def integrator(self, ocp, nlp) -> list:
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

        nlp.dynamics = nlp.ode_solver.integrator(ocp, nlp)
        if len(nlp.dynamics) == 1:
            if ocp.n_threads > 1:
                nlp.par_dynamics = nlp.dynamics[0].map(nlp.ns, "thread", ocp.n_threads)
            nlp.dynamics = nlp.dynamics * nlp.ns


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

    def integrator(self, ocp, nlp) -> list:
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

        ode_opt = {
            "t0": 0,
            "tf": nlp.dt,
            "model": nlp.model,
            "param": nlp.p,
            "param_scaling": nlp.p_scaling,
            "cx": nlp.cx,
            "idx": 0,
            "control_type": nlp.control_type,
            "number_of_finite_elements": self.steps,
        }
        ode = {"x": nlp.x, "p": nlp.u, "ode": nlp.dynamics_func}

        if nlp.external_forces:
            dynamics_out = []
            for idx in range(len(nlp.external_forces)):
                ode_opt["idx"] = idx
                dynamics_out.append(nlp.ode_solver.rk_integrator(ode, ode_opt))
            return dynamics_out
        else:
            return [nlp.ode_solver.rk_integrator(ode, ode_opt)]


class OdeSolver:
    """
    The public interface to the different OdeSolvers
    """

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

    class IRK(OdeSolverBase):
        """
        An implicit Runge-Kutta solver

        Attributes
        ----------
        polynome_degree: int
            The degree of the implicit RK

        Methods
        -------
        integrator(self, ocp, nlp) -> list
            The interface of the OdeSolver to the corresponding integrator
        """

        def __init__(self, polynome_degree: int = 4):
            """
            Parameters
            ----------
            polynome_degree: int
                The degree of the implicit RK
            """

            super(OdeSolver.IRK, self).__init__()
            self.polynome_degree = polynome_degree
            self.rk_integrator = IRK

        def integrator(self, ocp, nlp) -> list:
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

            if ocp.cx is casadi.SX:
                raise NotImplementedError("use_sx=True and OdeSolver.IRK are not yet compatible")

            if nlp.model.nbQuat() > 0:
                raise NotImplementedError(
                    "Quaternions can't be used with IRK yet. If you get this error, please notify the "
                    "developers and ping EveCharbie"
                )

            ode = {"x": nlp.x, "p": nlp.u, "ode": nlp.dynamics_func}
            ode_opt = {
                "t0": 0,
                "tf": nlp.dt,
                "model": nlp.model,
                "param": nlp.p,
                "param_scaling": nlp.p_scaling,
                "cx": nlp.cx,
                "idx": 0,
                "control_type": nlp.control_type,
                "irk_polynomial_interpolation_degree": self.polynome_degree,
            }
            return [nlp.ode_solver.rk_integrator(ode, ode_opt)]

    class CVODES(OdeSolverBase):
        """
        An interface to CVODES
        """

        def __init__(self):
            super(OdeSolver.CVODES, self).__init__()

        def integrator(self, ocp, nlp) -> list:
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

            if not isinstance(ocp.cx(), casadi.MX):
                raise RuntimeError("CVODES integrator can only be used with MX graphs")
            if len(ocp.v.params.size) != 0:
                raise RuntimeError("CVODES cannot be used while optimizing parameters")
            if nlp.external_forces:
                raise RuntimeError("CVODES cannot be used with external_forces")
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                raise RuntimeError("CVODES cannot be used with piece-wise linear controls (only RK4)")
            if not isinstance(nlp.ode_solver, OdeSolver.RK4):
                raise RuntimeError("CVODES is only implemented with RK4")

            ode = {"x": nlp.x, "p": nlp.u, "ode": nlp.dynamics_func(nlp.x, nlp.u, nlp.p)}
            ode_opt = {"t0": 0, "tf": nlp.dt, "number_of_finite_elements": nlp.ode_solver.steps}

            return [casadi.integrator("integrator", "cvodes", ode, ode_opt)]
