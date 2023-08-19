from typing import Callable, Any

import biorbd_casadi as biorbd
from biorbd_casadi import (
    GeneralizedCoordinates,
    GeneralizedVelocity,
    GeneralizedTorque,
    GeneralizedAcceleration,
)
from casadi import SX, MX, vertcat, horzcat, norm_fro, sin, cos, inv, jacobian, Function
import numpy as np

from ..misc.utils import check_version
from ..limits.path_conditions import Bounds
from ..misc.mapping import BiMapping, BiMappingList
from .biorbd_model import BiorbdModel

check_version(biorbd, "1.9.9", "1.10.0")



class StochasticBiorbdModel(BiorbdModel):
    """
    This class allows to define a biorbd model.
    """

    def __init__(self, bio_model: str | biorbd.Model, sensory_noise, motor_noise, force_field):

        super().__init__(bio_model.model)


        self.motor_noise = motor_noise
        self.sensory_noise = sensory_noise
        self.friction = np.array([[0.05, 0.025], [0.025, 0.05]])
        self.force_field_magnitude = force_field
        # jacobian_qddot_motor_noise = jacobian(self.stochastic_dynamics(q, qdot, tau, w), motor_noise)
        # jacobian_qddot_sensory_noise = (jacobian(self.stochastic_dynamics(q, qdot, tau, w), sensory_noise)
        # self._jacobian_qddot_noise = Function(...)

    def stochastic_dynamics(self, q, qdot, tau, ref, k, sensory_noise=None, motor_noise=None, with_gains=True):
        """ note: it should be called feedback dynamics ?? """
        if sensory_noise is None:
            sensory_noise = self.sensory_noise
        if motor_noise is None:
            motor_noise = self.motor_noise

        from bioptim.optimization.optimization_variable import OptimizationVariable  # Avoid circular import
        k_matrix = OptimizationVariable.reshape_sym_to_matrix(k, self.nb_tau, self.nb_q + self.nb_qdot)

        tau_fb = tau

        if with_gains:

            hand_pos = self.markers(q)[2][:2]
            hand_vel = self.marker_velocities(q, qdot)[2][:2]
            end_effector = vertcat(hand_pos, hand_vel)

            tau_fb += self.get_excitation_with_feedback(k_matrix, end_effector, ref, sensory_noise)

        tau_force_field = self.get_force_field(q)

        torques_computed = tau_fb + motor_noise + tau_force_field

        mass_matrix = self.mass_matrix(q)
        non_linear_effects = self.non_linear_effects(q, qdot)

        return inv(mass_matrix) @ (torques_computed - non_linear_effects - self.friction @ qdot)

    def get_excitation_with_feedback(self, k_matrix, hand_pos_velo, ref, sensory_noise):
        """
        Get the effect of the feedback.

        Parameters
        ----------
        k_matrix: MX.sym
            The feedback gains
        hand_pos_velo: MX.sym
            The position and velocity of the hand
        ref: MX.sym
            The reference position and velocity of the hand
        sensory_noise: MX.sym
            The sensory noise
        """
        return k_matrix @ ((hand_pos_velo - ref) + sensory_noise)


    def get_force_field(self, q):
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
        f_force_field = self.force_field_magnitude * (l1 * cos(q[0]) + l2 * cos(q[0] + q[1]))
        hand_pos = MX(2, 1)
        hand_pos[0] = l2 * sin(q[0] + q[1]) + l1 * sin(q[0])
        hand_pos[1] = l2 * sin(q[0] + q[1])
        tau_force_field = -f_force_field @ hand_pos
        return tau_force_field

    def _initialize_stochastic_jacobian(self, stochastic_forward_dynamics):
        sensory_noise_sym = MX.sym("sensory_noise", self.nb_q + self.nb_qdot)
        motor_noise_sym = MX.sym("motor_noise", self.nb_tau)
        q_sym = MX.sym("q", self.nb_q)
        qdot_sym = MX.sym("qdot", self.nb_qdot)
        tau_sym = MX.sym("tau", self.nb_tau)
        ref = MX.sym("ref", self.nb_q + self.nb_qdot)
        k = MX.sym("k", (self.nb_q + self.nb_qdot) * 2, 1)

        qddot = self.stochastic_dynamics(q_sym, qdot_sym, tau_sym, ref, k, sensory_noise_sym, motor_noise_sym)

        jacobian_qddot_noise = jacobian(qddot, vertcat(sensory_noise_sym, motor_noise_sym))

        temporary_jacobian_qddot_noise_func = Function(
            "jacobian_qddot_noise",
            [q_sym, qdot_sym, tau_sym, ref, k, sensory_noise_sym, motor_noise_sym],
            [jacobian_qddot_noise]
        )

        # todo: to verify, seems good but still
        self._jacobian_qddot_noise = Function(
            "jacobian_qddot_noise",
            [q_sym, qdot_sym, tau_sym, ref, k],
            [
                temporary_jacobian_qddot_noise_func(
                    q_sym,
                    qdot_sym,
                    tau_sym,
                    ref,
                    k,
                    self.sensory_noise,
                    self.motor_noise
                )
            ]
        )

    def stochastic_jacobian(self, q, qdot, tau, ref, k):
        self._initialize_stochastic_jacobian()
        return self._jacobian_qddot_noise(q, qdot, tau, ref, k)

    def next

    def _initialize_stochastic_jacobian(self, stochastic_forward_dynamics):
        sensory_noise_sym = MX.sym("sensory_noise", self.nb_q + self.nb_qdot)
        motor_noise_sym = MX.sym("motor_noise", self.nb_tau)
        q_sym = MX.sym("q", self.nb_q)
        qdot_sym = MX.sym("qdot", self.nb_qdot)
        tau_sym = MX.sym("tau", self.nb_tau)
        ref = MX.sym("ref", self.nb_q + self.nb_qdot)
        k = MX.sym("k", (self.nb_q + self.nb_qdot) * 2, 1)

        qddot = self.stochastic_dynamics(q_sym, qdot_sym, tau_sym, ref, k, sensory_noise_sym, motor_noise_sym)

        jacobian_qddot_noise = jacobian(qddot, vertcat(sensory_noise_sym, motor_noise_sym))

        temporary_jacobian_qddot_noise_func = Function(
            "jacobian_qddot_noise",
            [q_sym, qdot_sym, tau_sym, ref, k, sensory_noise_sym, motor_noise_sym],
            [jacobian_qddot_noise]
        )

        # todo: to verify, seems good but still
        self._jacobian_qddot_noise = Function(
            "jacobian_qddot_noise",
            [q_sym, qdot_sym, tau_sym, ref, k],
            [
                temporary_jacobian_qddot_noise_func(
                    q_sym,
                    qdot_sym,
                    tau_sym,
                    ref,
                    k,
                    self.sensory_noise,
                    self.motor_noise
                )
            ]
        )

    def stochastic_jacobian(self, q, qdot, tau, ref, k):
        self._initialize_stochastic_jacobian()
        return self._jacobian_qddot_noise(q, qdot, tau, ref, k)
