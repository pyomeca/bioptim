from bioptim import Solver
from bioptim.misc.enums import SolverType


def test_ipopt_solver_options():
    solver = Solver.IPOPT()
    assert solver.type == SolverType.IPOPT
    assert solver.show_online_optim is False
    assert solver.show_options is None
    assert solver.tol == 1e-6
    assert solver.dual_inf_tol == 1.0
    assert solver.constr_viol_tol == 0.0001
    assert solver.compl_inf_tol == 0.0001
    assert solver.acceptable_tol == 1e-6
    assert solver.acceptable_dual_inf_tol == 1e10
    assert solver.acceptable_constr_viol_tol == 1e-2
    assert solver.acceptable_compl_inf_tol == 1e-2
    assert solver.max_iter == 1000
    assert solver.hessian_approximation == "exact"
    assert solver.limited_memory_max_history == 50
    assert solver.linear_solver == "mumps"
    assert solver.mu_init == 0.1
    assert solver.warm_start_init_point == "no"
    assert solver.warm_start_mult_bound_push == 0.001
    assert solver.warm_start_slack_bound_push == 0.001
    assert solver.warm_start_bound_push == 0.001
    assert solver.warm_start_slack_bound_frac == 0.001
    assert solver.warm_start_bound_frac == 0.001
    assert solver.bound_push == 0.01
    assert solver.bound_frac == 0.01
    assert solver.print_level == 5
    assert solver.c_compile is False
