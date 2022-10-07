import numpy as np

from .interface_utils import (
    generic_online_optim,
    generic_solve,
    generic_dispatch_bounds,
    generic_dispatch_obj_func,
    generic_get_all_penalties,
    generic_set_lagrange_multiplier,
)
from .solver_interface import SolverInterface
from ..interfaces.solver_options import Solver
from ..limits.path_conditions import Bounds
from ..limits.phase_transition import PhaseTransitionFcn
from ..misc.enums import InterpolationType, ControlType, Node, SolverType, IntegralApproximation
from ..optimization.solution import Solution
from ..optimization.non_linear_program import NonLinearProgram
from ..misc.enums import (
    SolverType,
)


class IpoptInterface(SolverInterface):
    """
    The Ipopt solver interface

    Attributes
    ----------
    options_common: dict
        Options irrelevant of a specific ocp
    opts: IPOPT
        Options of the current ocp
    ipopt_nlp: dict
        The declaration of the variables Ipopt-friendly
    ipopt_limits: dict
        The declaration of the bound Ipopt-friendly
    lam_g: np.ndarray
        The lagrange multiplier of the constraints to initialize the solver
    lam_x: np.ndarray
        The lagrange multiplier of the variables to initialize the solver

    Methods
    -------
    online_optim(self, ocp: OptimalControlProgram)
        Declare the online callback to update the graphs while optimizing
    solve(self) -> dict
        Solve the prepared ocp
    set_lagrange_multiplier(self, sol: dict)
        Set the lagrange multiplier from a solution structure
    __dispatch_bounds(self)
        Parse the bounds of the full ocp to a Ipopt-friendly one
    __dispatch_obj_func(self)
        Parse the objective functions of the full ocp to a Ipopt-friendly one
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        """

        super().__init__(ocp)

        self.options_common = {}
        self.opts = Solver.IPOPT()
        self.solver_name = SolverType.IPOPT.value

        self.ipopt_nlp = {}
        self.ipopt_limits = {}
        self.ocp_solver = None
        self.c_compile = False

        self.lam_g = None
        self.lam_x = None

    def online_optim(self, ocp, show_options: dict = None):
        """
        Declare the online callback to update the graphs while optimizing

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the current OptimalControlProgram
        show_options: dict
            The options to pass to PlotOcp
        """

        generic_online_optim(self, ocp, show_options)

    def solve(self) -> dict:
        """
        Solve the prepared ocp

        Returns
        -------
        A reference to the solution
        """
        return generic_solve(self)

    def set_lagrange_multiplier(self, sol: Solution):
        """
        Set the lagrange multiplier from a solution structure

        Parameters
        ----------
        sol: dict
            A solution structure where the lagrange multipliers are set
        """
        sol = generic_set_lagrange_multiplier(self, sol)

    def dispatch_bounds(self):
        """
        Parse the bounds of the full ocp to a Ipopt-friendly one
        """
        return generic_dispatch_bounds(self)

    def dispatch_obj_func(self):
        """
        Parse the objective functions of the full ocp to a Ipopt-friendly one

        Returns
        -------
        Union[SX, MX]
            The objective function
        """
        return generic_dispatch_obj_func(self)

    def get_all_penalties(self, nlp: NonLinearProgram, penalties):
        """
        Parse the penalties of the full ocp to a Ipopt-friendly one

        Parameters
        ----------
        nlp: NonLinearProgram
            The nonlinear program to parse the penalties from
        penalties:
            The penalties to parse
        Returns
        -------

        """
        return generic_get_all_penalties(self, nlp, penalties)

        def format_target(target_in: np.array) -> np.array:
            """
            Format the target of a penalty to a numpy array

            Parameters
            ----------
            target_in: np.array
                The target of the penalty
            Returns
            -------
                np.array
                    The target of the penalty formatted to a numpy array
            """
            if len(target_in.shape) == 2:
                target_out = target_in[:, penalty.node_idx.index(idx)]
            elif len(target_in.shape) == 3:
                target_out = target_in[:, :, penalty.node_idx.index(idx)]
            else:
                raise NotImplementedError("penalty target with dimension != 2 or 3 is not implemented yet")
            return target_out

        def get_x_and_u_at_idx(_penalty, _idx):
            if _penalty.transition:
                ocp = self.ocp
                _x = vertcat(ocp.nlp[_penalty.phase_pre_idx].X[-1], ocp.nlp[_penalty.phase_post_idx].X[0][:, 0])
                _u = vertcat(ocp.nlp[_penalty.phase_pre_idx].U[-1], ocp.nlp[_penalty.phase_post_idx].U[0])
            elif _penalty.multinode_constraint:
                ocp = self.ocp
                _x = vertcat(
                    ocp.nlp[_penalty.phase_first_idx].X[_penalty.node_idx[0]],
                    ocp.nlp[_penalty.phase_second_idx].X[_penalty.node_idx[1]][:, 0],
                )
                # Make an exception to the fact that U is not available for the last node
                mod_u0 = 1 if _penalty.first_node == Node.END else 0
                mod_u1 = 1 if _penalty.second_node == Node.END else 0
                _u = vertcat(
                    ocp.nlp[_penalty.phase_first_idx].U[_penalty.node_idx[0] - mod_u0],
                    ocp.nlp[_penalty.phase_second_idx].U[_penalty.node_idx[1] - mod_u1],
                )
            elif _penalty.integrate:
                _x = nlp.X[_idx]
                _u = nlp.U[_idx][:, 0] if _idx < len(nlp.U) else []
            else:
                _x = nlp.X[_idx][:, 0]
                _u = nlp.U[_idx][:, 0] if _idx < len(nlp.U) else []

            if _penalty.derivative or _penalty.explicit_derivative:
                _x = horzcat(_x, nlp.X[_idx + 1][:, 0])
                _u = horzcat(
                    _u, nlp.U[_idx + 1][:, 0] if _idx + 1 < len(nlp.U) else []
                )

            if _penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL:
                _x = horzcat(_x, nlp.X[_idx + 1][:, 0])
                if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    _u = horzcat(
                        _u, nlp.U[_idx + 1][:, 0] if _idx + 1 < len(nlp.U) else []
                    )

            if _penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
                if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    _u = horzcat(
                        _u, nlp.U[_idx + 1][:, 0] if _idx + 1 < len(nlp.U) else []
                    )
            return _x, _u

        param = self.ocp.cx(self.ocp.v.parameters_in_list.cx)
        out = self.ocp.cx()
        for penalty in penalties:
            if not penalty:
                continue

            if penalty.multi_thread:
                if penalty.target is not None and len(penalty.target[0].shape) != 2:
                    raise NotImplementedError(
                        "multi_thread penalty with target shape != [n x m] is not implemented yet"
                    )
                target = penalty.target if penalty.target is not None else []

                x = nlp.cx()
                u = nlp.cx()
                for idx in penalty.node_idx:
                    x_tp, u_tp = get_x_and_u_at_idx(penalty, idx)
                    x = horzcat(x, x_tp)
                    u = horzcat(u, u_tp)
                if (
                    penalty.derivative or penalty.explicit_derivative or penalty.node[0] == Node.ALL
                ) and nlp.control_type == ControlType.CONSTANT:
                    u = horzcat(u, u[:, -1])

                p = reshape(penalty.weighted_function(x, u, param, penalty.weight, target, penalty.dt), -1, 1)

            else:
                p = self.ocp.cx()
                for idx in penalty.node_idx:
                    if penalty.target is None:
                        target = []
                    elif (
                        penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL
                        or penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                    ):
                        target0 = format_target(penalty.target[0])
                        target1 = format_target(penalty.target[1])
                        target = np.vstack((target0, target1)).T
                    else:
                        target = format_target(penalty.target[0])

                    if np.isnan(np.sum(target)):
                        continue

                    if not nlp:
                        x = []
                        u = []
                    else:
                        x, u = get_x_and_u_at_idx(penalty, idx)
                    p = vertcat(p, penalty.weighted_function(x, u, param, penalty.weight, target, penalty.dt))
            out = vertcat(out, sum2(p))
        return out
