
import os
import pytest
import numpy as np
from casadi import MX
import biorbd_casadi as biorbd
from bioptim import (
    MultiBiorbdModel,
)


def test_biorbd_model_import():
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = "/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = "/models/triple_pendulum_modified_inertia.bioMod"
    MultiBiorbdModel((bioptim_folder + biorbd_model_path, bioptim_folder + biorbd_model_path_modified_inertia))

    MultiBiorbdModel((biorbd.Model(bioptim_folder + biorbd_model_path), biorbd.Model(bioptim_folder + biorbd_model_path_modified_inertia)))

    with pytest.raises(RuntimeError, match="Type must be a tuple"):
        MultiBiorbdModel(1)


# Test the functions from the class MultiBiorbdModel
def test_biorbd_model():
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = "/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = "/models/triple_pendulum_modified_inertia.bioMod"
    models = MultiBiorbdModel((bioptim_folder + biorbd_model_path, bioptim_folder + biorbd_model_path_modified_inertia))

    # replace the method called by the functions from the class MultiBiorbdModel
    model_deep_copied = models.deep_copy()
    model_copied = models.copy()
    model_serialized = models.serialize()
    model_gravity = models.gravity()
    model_set_gravity = models.set_gravity(np.array([0, 0, -9.81]), 0)
    nb_tau = models.nb_tau()
    nb_segments = models.nb_segments()
    segment_index = models.segment_index("Seg1_1", 0)
    nb_quaternions = models.nb_quaternions()
    nb_q = models.nb_q()
    nb_qdot = models.nb_qdot()
    nb_qddot = models.nb_qddot()
    nb_root = models.nb_root()
    segments = models.segments()
    q = MX(nb_q)
    qdot = MX(nb_q)
    tau = MX(nb_tau)
    qddot_joints = MX(nb_tau - nb_root)
    f_ext = MX(6)
    f_contact = MX(6)
    homogeneous_matrices_in_global = models.homogeneous_matrices_in_global(q, 0, 0)
    homogeneous_matrices_in_child = models.homogeneous_matrices_in_child(0)
    mass = models.mass()
    center_of_mass = models.center_of_mass()
    center_of_mass_velocity = models.center_of_mass_velocity(q, qdot, 0)
    center_of_mass_acceleration = models.center_of_mass_acceleration(q, qdot, qdot, 0)
    angular_momentum = models.angular_momentum(q, qdot, 0)
    reshape_qdot = models.reshape_qdot(q, qdot, 1)
    segment_angular_velocity = models.segment_angular_velocity(q, qdot, 0)
    name_dof = models.name_dof()
    contact_names = models.contact_names()
    nb_soft_contacts = models.nb_soft_contacts()
    soft_contact_names = models.soft_contact_names()
    soft_contact = models.soft_contact()
    muscle_names = models.muscle_names()
    nb_muscles = models.nb_muscles()
    torque = models.torque(tau, q, qdot)
    forward_dynamics_free_floating_base = models.forward_dynamics_free_floating_base(q, qdot, qddot_joints)
    forward_dynamics = models.forward_dynamics(q, qdot, tau)
    constrained_forward_dynamics = models.constrained_forward_dynamics(q, qdot, tau, f_ext)
    inverse_dynamics = models.inverse_dynamics(q, qdot, tau, f_ext, f_contact)
    contact_forces_from_constrained_dynamics = models.contact_forces_from_constrained_dynamics(q, qdot, tau, f_ext)
    qdot_from_inpact = models.qdot_from_impact(q, qdot, qdot)


