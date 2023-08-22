"""
This file contains the functions that are common for multiple stochastic examples.
"""

import casadi as cas
from bioptim import NoiseType, StochasticBioModel, DynamicsFunctions


def dynamics_torque_driven_with_feedbacks(states, controls, parameters, stochastic_variables, nlp, noise_type: NoiseType, with_gains=True):

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
    k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
    k_matrix = StochasticBioModel.reshape_sym_to_matrix(k, nlp.model.matrix_shape_k)

    if noise_type == NoiseType.NONE:
        motor_noise = 0
        sensory_noise = 0
    # elif noise_type == NoiseType.MAGNITUDE:
    #     motor_noise = self.motor_noise_magnitude
    #     sensory_noise = self.sensory_noise_magnitude
    elif noise_type == NoiseType.SYMBOLIC:
        motor_noise = nlp.model.motor_noise_sym
        sensory_noise = nlp.model.sensory_noise_sym
    else:
        ValueError("Wrong noise_type")

    tau_fb = tau

    if with_gains:
        end_effector = nlp.model.sensory_reference(states, controls, parameters, stochastic_variables, nlp)
        tau_fb += get_excitation_with_feedback(k_matrix, end_effector, ref, sensory_noise)

    tau_force_field = get_force_field(q, nlp.model.force_field_magnitude)

    torques_computed = tau_fb + motor_noise + tau_force_field

    mass_matrix = nlp.model.mass_matrix(q)
    non_linear_effects = nlp.model.non_linear_effects(q, qdot)

    return cas.inv(mass_matrix) @ (torques_computed - non_linear_effects - nlp.model.friction_coefficients @ qdot)

def get_force_field(q, force_field_magnitude):
    """
    Get the effect of the force field.

    Parameters
    ----------
    q: MX.sym
        The generalized coordinates
    force_field_magnitude: float
        The magnitude of the force field
    """
    l1 = 0.3
    l2 = 0.33
    f_force_field = force_field_magnitude * (l1 * cas.cos(q[0]) + l2 * cas.cos(q[0] + q[1]))
    hand_pos = cas.MX(2, 1)
    hand_pos[0] = l2 * cas.sin(q[0] + q[1]) + l1 * cas.sin(q[0])
    hand_pos[1] = l2 * cas.sin(q[0] + q[1])
    tau_force_field = -f_force_field @ hand_pos
    return tau_force_field


def get_excitation_with_feedback(k, hand_pos_velo, ref, sensory_noise):
    """
    Get the effect of the feedback.

    Parameters
    ----------
    k: MX.sym
        The feedback gains
    hand_pos_velo: MX.sym
        The position and velocity of the hand
    ref: MX.sym
        The reference position and velocity of the hand
    sensory_noise: MX.sym
        The sensory noise
    """
    return k @ ((hand_pos_velo - ref) + sensory_noise)


