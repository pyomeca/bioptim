from typing import Callable
from casadi import MX, SX

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

    sensory_reference_function: Callable
    motor_noise_mapping: BiMappingList

    def stochastic_dynamics(self, q, qdot, tau, ref, k, symbolic_noise=False, with_gains=True):
        """The stochastic dynamics that should be applied to the model"""

    @staticmethod
    def reshape_to_matrix(var, shape):
        """
        Restore the matrix form of the variables
        """
        shape_0, shape_1 = shape
        matrix = MX(shape_0, shape_1)
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
        matrix = MX(shape_0, shape_1)
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
        matrix = MX.zeros(shape_0, shape_0)
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
        matrix = MX.zeros(shape_0, shape_0)
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
        vector = MX.zeros(shape_0 * shape_1)
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                vector[shape_0 * s1 + s0] = matrix[s0, s1]
        return vector
