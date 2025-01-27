from typing import Callable
import numpy as np

from ..protocols.biomodel import BioModel
from ...misc.mapping import BiMappingList


class StochasticBioModel(BioModel):
    """
    This class allows to define a model that can be used in a stochastic optimal control problem.
    """

    sensory_noise_magnitude: np.ndarray
    motor_noise_magnitude: np.ndarray

    sensory_reference: Callable
    motor_noise_mapping: BiMappingList

    matrix_shape_k: tuple[int, int]
    matrix_shape_c: tuple[int, int]
    matrix_shape_a: tuple[int, int]
    matrix_shape_cov: tuple[int, int]
    matrix_shape_cov_cholesky: tuple[int, int]
    matrix_shape_m: tuple[int, int]

    def stochastic_dynamics(self, q, qdot, tau, ref, k, with_noise=True):
        """The stochastic dynamics that should be applied to the model"""

    def compute_torques_from_noise_and_feedback(
        self, nlp, time, states, controls, parameters, algebraic_states, sensory_noise, motor_noise
    ):
        """Compute the torques from the noises, feedbacks and feedforwards"""

    def sensory_reference(self, time, states, controls, parameters, algebraic_states, nlp):
        """Compute the sensory reference"""

    @staticmethod
    def reshape_to_matrix(var, shape):
        """
        Restore the matrix form of the variables

        See Also
        --------
        reshape_to_vector
        """

        if var.shape[0] != shape[0] * shape[1]:
            raise RuntimeError(f"Cannot reshape: the variable shape is {var.shape} and the expected shape is {shape}")

        shape_0, shape_1 = shape
        if isinstance(var, np.ndarray):
            matrix = np.zeros((shape_0, shape_1))
        else:
            matrix = type(var).zeros(shape_0, shape_1)
        for s0 in range(shape_1):
            for s1 in range(shape_0):
                matrix[s1, s0] = var[s0 * shape_0 + s1]
        return matrix

    @staticmethod
    def reshape_to_cholesky_matrix(var, shape):
        """
        Restore the lower diagonal matrix form of the variables vector
        """
        shape_0, _ = shape
        matrix = type(var).zeros(shape_0, shape_0)

        i = 0
        for s0 in range(shape_0):
            for s1 in range(s0 + 1):
                matrix[s1, s0] = var[i]
                i += 1

        if var.shape[0] != i:
            raise RuntimeError(
                f"Cannot reshape: the variable shape is {var.shape} and the expected shape is the number of element in a triangular matrix of size {shape_0}, which means {i} elements"
            )

        i = 0
        for s0 in range(shape_0):
            for s1 in range(s0 + 1):
                matrix[s1, s0] = var[i]
                i += 1
        return matrix

    @staticmethod
    def reshape_to_vector(matrix):
        """
        Restore the vector form of the matrix
        
        See Also
        --------
        reshape_to_matrix
        """
        shape_0, shape_1 = matrix.shape[0], matrix.shape[1]
        if isinstance(matrix, np.ndarray):
            vector = np.zeros(shape_0 * shape_1)
        else:
            vector = type(matrix).zeros(shape_0 * shape_1)

        for s0 in range(shape_0):
            for s1 in range(shape_1):
                vector[shape_0 * s1 + s0] = matrix[s0, s1]
        return vector
