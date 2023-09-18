from enum import Enum
from typing import Callable
from casadi import MX
import numpy as np

from .biomodel import BioModel
from ..misc.mapping import BiMappingList


class StochasticBioModel(BioModel):
    """
    This class allows to define a model that can be used in a stochastic optimal control problem.
    """

    sensory_noise_magnitude: float
    motor_noise_magnitude: float

    sensory_noise_sym: MX.sym
    motor_noise_sym: MX.sym

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

    @staticmethod
    def reshape_to_matrix(var, shape):
        """
        Restore the matrix form of the variables
        """
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
    def reshape_sym_to_matrix(var, shape):
        """
        Restore the matrix form of the variables
        """

        shape_0, shape_1 = shape
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
        return matrix

    @staticmethod
    def reshape_sym_to_cholesky_matrix(var, shape):
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
        return matrix

    @staticmethod
    def reshape_to_vector(matrix):
        """
        Restore the vector form of the matrix
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
