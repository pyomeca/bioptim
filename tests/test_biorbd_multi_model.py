
import os
import pytest
import numpy as np
from casadi import MX, vertcat
import biorbd_casadi as biorbd
from bioptim import (
    MultiBiorbdModel,
    BiMapping,
    BiMappingList,
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


    nb_q = models.nb_q()
    nb_qdot = models.nb_qdot()
    nb_qddot = models.nb_qddot()
    nb_root = models.nb_root()
    nb_tau = models.nb_tau()
    nb_quaternions = models.nb_quaternions()
    nb_segments = models.nb_segments()
    nb_muscles = models.nb_muscles()
    nb_soft_contacts = models.nb_soft_contacts()
    nb_markers = models.nb_markers()
    nb_rigid_contacts = models.nb_rigid_contacts()
    nb_contacts = models.nb_contacts()
    nb_dof = models.nb_dof()

    name_dof = models.name_dof()
    contact_names = models.contact_names()
    soft_contact_names = models.soft_contact_names()
    marker_names = models.marker_names()
    muscle_names = models.muscle_names()

    q_mapping = BiMapping('q', [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    qdot_mapping = BiMapping('qdot', [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    qddot_mapping = BiMapping('qddot', [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    tau_mapping = BiMapping('tau', [None, 0, 1, None, 0, 2], [1, 2, 5])
    states_mapping = BiMappingList()
    states_mapping.add(q_mapping)
    states_mapping.add(qdot_mapping)

    q = MX(np.random(nb_q, seed=42))
    qdot = MX(nb_q)
    tau = MX(nb_tau)
    qddot_joints = MX(nb_tau - nb_root)
    f_ext = MX(6)
    f_contact = MX(6)
    muscle_excitations = MX(nb_muscles)

    model_deep_copied = models.deep_copy()
    model_copied = models.copy()
    model_serialized = models.serialize()
    model_gravity = models.gravity()
    model_set_gravity = models.set_gravity(np.array([0, 0, -9.81]), 0)
    segment_index = models.segment_index("Seg1_1", 0)
    segments = models.segments()
    homogeneous_matrices_in_global = models.homogeneous_matrices_in_global(q, 0, 0)
    homogeneous_matrices_in_child = models.homogeneous_matrices_in_child(0)
    mass = models.mass()
    center_of_mass = models.center_of_mass()
    center_of_mass_velocity = models.center_of_mass_velocity(q, qdot, 0)
    center_of_mass_acceleration = models.center_of_mass_acceleration(q, qdot, qdot, 0)
    angular_momentum = models.angular_momentum(q, qdot, 0)
    reshape_qdot = models.reshape_qdot(q, qdot, 1)
    segment_angular_velocity = models.segment_angular_velocity(q, qdot, 0)
    soft_contact = models.soft_contact()
    torque = models.torque(tau, q, qdot)
    forward_dynamics_free_floating_base = models.forward_dynamics_free_floating_base(q, qdot, qddot_joints)
    forward_dynamics = models.forward_dynamics(q, qdot, tau)
    constrained_forward_dynamics = models.constrained_forward_dynamics(q, qdot, tau, f_ext)
    inverse_dynamics = models.inverse_dynamics(q, qdot, tau, f_ext, f_contact)
    contact_forces_from_constrained_dynamics = models.contact_forces_from_constrained_dynamics(q, qdot, tau, f_ext)
    qdot_from_inpact = models.qdot_from_impact(q, qdot, qdot)
    muscle_activation_dot = models.muscle_activation_dot(muscle_excitations)
    muscle_joint_torque = models.muscle_joint_torque(muscle_excitations, q, qdot)
    markers = models.markers(q)
    marker = models.marker(q, index=0, model_index=0, reference_segment_index=0)
    marker_index = models.marker_index("Marker_1", 0)
    marker_velocities = models.marker_velocities(q, qdot, reference_index=0)
    tau_max = models.tau_max(q, qdot)
    # rigid_contact_acceleration = models.rigid_contact_acceleration(q, qdot, qddot, 0) # to be added when the code works
    soft_contact_forces = models.soft_contact_forces(q, qdot)
    reshape_fext_to_fcontact = models.reshape_fext_to_fcontact(f_ext)
    normalize_state_quaternions = models.normalize_state_quaternions(vertcat(q, qdot))
    get_quaternion_idx = models.get_quaternion_idx()
    contact_forces = models.contact_forces(q, qdot, tau, f_ext)
    passive_joint_torque = models.passive_joint_torque(q, qdot)
    q_mapping = models._q_mapping(q_mapping)
    qdot_mapping = models._q_mapping(qdot_mapping)
    qddot_mapping = models._q_mapping(qddot_mapping)
    tau_mapping = models._q_mapping(tau_mapping)
    bounds_from_ranges = models.bounds_from_ranges(['q', 'qdot'], states_mapping)

