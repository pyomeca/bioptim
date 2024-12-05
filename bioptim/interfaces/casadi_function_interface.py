from abc import ABC, abstractmethod

from casadi import Callback, Function, Sparsity, DM, MX, jacobian
import numpy as np


class CasadiFunctionInterface(Callback, ABC):
    def __init__(self, name: str, opts={}):
        self.reverse_function = None

        super(CasadiFunctionInterface, self).__init__()
        self.construct(name, opts)  # Defines the self.mx_in()
        self._cached_mx_in = super().mx_in()

    @abstractmethod
    def inputs_len(self) -> list[int]:
        """
        The len of the inputs of the function. This will help create the MX/SX vectors such that each element of the list
        is the length of the input vector (i.e. the sparsity of the input vector).

        Example:
        def inputs_len(self) -> list[int]:
            return [3, 4]  # Assuming two inputs x and y of length 3 and 4 respectively
        """
        pass

    @abstractmethod
    def outputs_len(self) -> list[int]:
        """
        The len of the outputs of the function. This will help create the MX/SX vectors such that each element of the list
        is the length of the output vector (i.e. the sparsity of the output vector).

        Example:
        def outputs_len(self) -> list[int]:
            return [5]  # Assuming the output is a 5x1 vector
        """
        pass

    @abstractmethod
    def function(self, *args) -> np.ndarray | DM:
        """
        The actual function to interface with casadi. The callable that returns should be callable by function(*mx_in).
        If your function needs more parameters, they should be encapsulated in a partial.

        Example:
        def function(self, x, y):
            x = np.array(x)[:, 0]
            y = np.array(y)[:, 0]
            return np.array(
                [
                    x[0] * y[1] + x[0] * y[0] * y[0],
                    x[1] * x[1] + 2 * y[1],
                    x[0] * x[1] * x[2],
                    x[2] * x[1] * y[2] + 2 * y[3] * y[2],
                    y[0] * y[1] * y[2] * y[3],
                ]
            )
        """
        pass

    @abstractmethod
    def jacobians(self, *args) -> list[np.ndarray | DM]:
        """
        All the jacobians evaluated at *args. Each of the jacobian should be of the shape (n_out, n_in), where n_out is
        the length of the output vector (the same for all) and n_in is the length of the input element (specific to each
        input element).

        Example:
        def jacobians(self, x, y):
            x = np.array(x)[:, 0]
            y = np.array(y)[:, 0]
            jacobian_x = np.array(
                [
                    [y[1] + y[0] * y[0], 0, 0],
                    [0, 2 * x[1], 0],
                    [x[1] * x[2], x[0] * x[2], x[0] * x[1]],
                    [0, x[2] * y[2], x[1] * y[2]],
                    [0, 0, 0],
                ]
            )
            jacobian_y = np.array(
                [
                    [x[0] * 2 * y[0], x[0], 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, x[1] * x[2] + 2 * y[3], 2 * y[2]],
                    [y[1] * y[2] * y[3], y[0] * y[2] * y[3], y[0] * y[1] * y[3], y[0] * y[1] * y[2]],
                ]
            )
            return [jacobian_x, jacobian_y]  # There are as many jacobians as there are inputs
        """
        pass

    def mx_in(self) -> MX:
        """
        Get the MX in, but it is ensured that the MX are the same at each call
        """
        return self._cached_mx_in

    def get_n_in(self):
        return len(self.inputs_len())

    def get_n_out(self):
        return len(self.outputs_len())

    def get_sparsity_in(self, i):
        return Sparsity.dense(self.inputs_len()[i], 1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(self.outputs_len()[i], 1)

    def eval(self, *args):
        return [self.function(*args[0])]

    def has_reverse(self, nadj):
        return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        class Reverse(Callback):
            def __init__(self, parent, jacobian_functions, opts={}):
                self._sparsity_in = parent.mx_in() + parent.mx_out()
                self._sparsity_out = parent.mx_in()

                self.jacobian_functions = jacobian_functions
                self.reverse_function = None
                Callback.__init__(self)
                self.construct("Reverse", opts)

            def get_n_in(self):
                return len(self._sparsity_in)

            def get_n_out(self):
                return len(self._sparsity_out)

            def get_sparsity_in(self, i):
                return Sparsity.dense(self._sparsity_in[i].shape)

            def get_sparsity_out(self, i):
                return Sparsity.dense(self._sparsity_out[i].shape)

            def eval(self, arg):
                # Find the index to evaluate from the last parameter which is a DM vector of 0s with one value being 1
                index = arg[-1].toarray()[:, 0].tolist().index(1.0)
                inputs = arg[:-1]
                return [jaco[index, :].T for jaco in self.jacobian_functions(*inputs)]

            def has_reverse(self, nadj):
                return nadj == 1

            def get_reverse(self, nadj, name, inames, onames, opts):
                if self.reverse_function is None:
                    self.reverse_function = Reverse(self, jacobian(self.jacobian_functions))

                cx_in = self.mx_in()
                nominal_out = self.mx_out()
                adj_seed = self.mx_out()
                return Function(name, cx_in + nominal_out + adj_seed, self.reverse_function.call(cx_in + adj_seed))

        # Package it in the [nominal_in + nominal_out + adj_seed] form that CasADi expects
        if self.reverse_function is None:
            self.reverse_function = Reverse(self, self.jacobians)

        cx_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return Function(name, cx_in + nominal_out + adj_seed, self.reverse_function.call(cx_in + adj_seed))
