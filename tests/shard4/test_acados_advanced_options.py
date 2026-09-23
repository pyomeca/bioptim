from bioptim import Solver


def test_acados_advanced_options_are_explicitly_exposed():
    solver = Solver.ACADOS()
    solver.set_integrator_type("ERK")
    solver.set_collocation_type("GAUSS_RADAU_IIA")
    solver.set_sim_method_num_stages(3)
    solver.set_sim_method_num_steps(2)
    solver.set_sim_method_newton_iter(7)
    solver.set_sim_method_newton_tol(1e-9)
    solver.set_sim_method_jac_reuse(1)
    solver.set_qp_solver("FULL_CONDENSING_HPIPM")
    solver.set_qp_solver_cond_N(5)
    solver.set_qp_solver_iter_max(75)
    solver.set_qp_solver_tolerances(1e-7)
    solver.set_regularize_method("CONVEXIFY")
    solver.set_levenberg_marquardt(1e-4)
    solver.set_globalization("MERIT_BACKTRACKING")

    options = solver.as_dict(None)
    assert options["integrator_type"] == "ERK"
    assert options["collocation_type"] == "GAUSS_RADAU_IIA"
    assert options["sim_method_num_stages"] == 3
    assert options["sim_method_num_steps"] == 2
    assert options["sim_method_newton_iter"] == 7
    assert options["sim_method_newton_tol"] == 1e-9
    assert options["sim_method_jac_reuse"] == 1
    assert options["qp_solver"] == "FULL_CONDENSING_HPIPM"
    assert options["qp_solver_cond_N"] == 5
    assert options["qp_solver_iter_max"] == 75
    assert options["qp_solver_tol_stat"] == 1e-7
    assert options["qp_solver_tol_eq"] == 1e-7
    assert options["qp_solver_tol_ineq"] == 1e-7
    assert options["qp_solver_tol_comp"] == 1e-7
    assert options["regularize_method"] == "CONVEXIFY"
    assert options["levenberg_marquardt"] == 1e-4
    assert options["globalization"] == "MERIT_BACKTRACKING"
    assert solver.only_first_options_has_changed


def test_acados_optional_options_are_not_forwarded_until_configured():
    options = Solver.ACADOS().as_dict(None)
    assert "sim_method_newton_tol" not in options
    assert "qp_solver_cond_N" not in options
    assert "qp_solver_tol_stat" not in options
