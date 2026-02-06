import pytest

import numpy as np
import numpy.testing as npt
from bioptim import OdeSolver, ControlType, PhaseDynamics, SolutionMerge, DefectType, ContactType, Solver

from ..utils import TestUtils


@pytest.mark.parametrize("control_type", [
    ControlType.CONSTANT,
    ControlType.LINEAR_CONTINUOUS,
])
@pytest.mark.parametrize(
    "defects_type",
    [
        DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS,
        DefectType.TAU_EQUALS_INVERSE_DYNAMICS,
    ],
)
@pytest.mark.parametrize("contact_types", [[ContactType.RIGID_EXPLICIT], [ContactType.RIGID_IMPLICIT]])
def test_contact_forces_inverse_dynamics_constraint_muscle(control_type, defects_type, contact_types):
    """
    Here we only test that the values of the objective and constraints at the initial guess are correct because the
    convergence of this problem is too long to wait for.
    Cannot have multi-node constraints in SHARED_DURING_THE_PHASE.
    ode_solver can only be COLLOCATION for this type of implicit constraints.
    """
    from bioptim.examples.toy_examples.muscle_driven_with_contact import (
        contact_forces_inverse_dynamics_constraint_muscle as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    if defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS and ContactType.RIGID_EXPLICIT in contact_types:
        with pytest.raises(
            NotImplementedError, match="Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet"
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/../../models/2segments_4dof_2contacts_1muscle.bioMod",
                phase_time=0.3,
                n_shooting=10,
                defects_type=defects_type,
                contact_types=contact_types,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        defects_type=defects_type,
        contact_types=contact_types,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve()

    # Check the values at the initial guess
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    g = np.array(sol.constraints)

    # TODO: TestUtils.assert_objective_value should be used, but it is broken here
    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
        if ContactType.RIGID_EXPLICIT in contact_types:
            npt.assert_almost_equal(f[0], 488.41630088)
            npt.assert_equal(g.shape, (326, 1))
            npt.assert_almost_equal(np.sum(g**2), 0)
        elif ContactType.RIGID_IMPLICIT in contact_types:
            npt.assert_almost_equal(f[0], 732.52924763)
            npt.assert_equal(g.shape, (443, 1))
            npt.assert_almost_equal(np.sum(g**2), 0)
        else:
            raise ValueError("Unexpected test configuration")
    elif defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        if ContactType.RIGID_EXPLICIT in contact_types:
            raise RuntimeError("Should be skipped above")
        elif ContactType.RIGID_IMPLICIT in contact_types:
            npt.assert_almost_equal(f[0], 732.52924762)
            npt.assert_equal(g.shape, (443, 1))
            npt.assert_almost_equal(np.sum(g**2), 0)
        else:
            raise ValueError("Unexpected test configuration")


    # Now test the internal constraints with random values
    g_internal = ocp.nlp[0].g_internal
    np.random.seed(0)
    t = 1.05
    dt = 0.1
    x = np.random.rand(40) * 0.1
    x_multi = np.random.rand(80) * 0.1
    u = np.random.rand(10) * 0.1
    u_multi = np.random.rand(20) * 0.1
    p = []
    a = np.random.rand(15) * 0.1
    a_multi = np.random.rand(30) * 0.1
    d = []


    # Since a small change in q has a large effect,the tolerance is quite high, but it does not mean these tests are not important !
    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
        if ContactType.RIGID_EXPLICIT in contact_types:
            # STATE_CONTINUITY (This is the important one since the contact are take into account in the dynamics)
            value = g_internal[0].function[3](t, dt, x, u, p, [], d)
            npt.assert_almost_equal(np.array(value).reshape(32, ), np.array(
                [-0.0771307, 0.0688179, 0.0100632, -0.0133405, 0.0524087, 0.00946086, 0.143732, 0.117869, 1.47277, -1.60227, 1.21719, 0.224233, 2.62202, 1.71563, -4.32498, -1.58506, -3.84431, 2.00027, -1.11185, 0.954616, 1.60419, -1.36516, -0.486072, 6.80542, 6.85567, -4.86582, -2.0102, -1.22047, -2.90894, 0.285695, -5.24616, -7.31138]),
                                    decimal=4)
        elif ContactType.RIGID_IMPLICIT in contact_types:
            # ALGEBRAIC_STATES_CONTINUITY_Multinode_P0n0_P0n1
            value = g_internal[6].function[6](t, dt, x_multi, u_multi, p, a_multi, d)
            npt.assert_almost_equal(np.array(value).reshape(3, ), np.array([-0.0629004, 0.0397554, 0.0231877]),
                                    decimal=4)

            # STATE_CONTINUITY (This is the important one since the defects are set in the dynamics)
            value = g_internal[9].function[3](t, dt, x, u, p, a, d)
            npt.assert_almost_equal(np.array(value).reshape(41, ), np.array(
                [-0.0771307, 0.0688179, 0.0100632, -0.0133405, 0.0524087, 0.00946086, 0.143732, 0.117869, 1.47277, -1.60227, 1.21719, 0.224233, 1.34074, 11.473, -1.79717, -7.85084, 0.466307, 1.5446, -0.505904, -3.84431, 2.00027, -1.11185, 0.954616, 0.355899, 8.39088, 1.94447, 0.772184, 1.36189, -1.38405, 0.797744, 6.85567, -4.86582, -2.0102, -1.22047, -3.13449, 10.1336, -4.82436, -11.2889, -5.53175, 0.248094, -4.12106]), decimal=4)

    elif defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        if ContactType.RIGID_IMPLICIT in contact_types:
            # ALGEBRAIC_STATES_CONTINUITY_Multinode_P0n0_P0n1
            value = g_internal[6].function[6](t, dt, x_multi, u_multi, p, a_multi, d)
            npt.assert_almost_equal(np.array(value).reshape(3, ), np.array([-0.0629004, 0.0397554, 0.0231877]),
                                    decimal=4)

            # STATE_CONTINUITY (This is the important one since the defects are set in the dynamics)
            value = g_internal[9].function[3](t, dt, x, u, p, a, d)
            npt.assert_almost_equal(np.array(value).reshape(41, ), np.array(
                [-0.0771307, 0.0688179, 0.0100632, -0.0133405, 0.0524087, 0.00946086, 0.143732, 0.117869, 1.47277, -1.60227, 1.21719, 0.224233, -11.6701, -153.657, -8.13758, -1.89798, 0.466307, 1.5446, -0.505904, -3.84431, 2.00027, -1.11185, 0.954616, -12.1213, -115.205, -7.96739, -1.63524, 1.36189, -1.38405, 0.797744, 6.85567, -4.86582, -2.0102, -1.22047, 60.6238, -134.478, 15.8761, -1.60747, -5.53175, 0.248094, -4.12106]),
                                    decimal=3)
