from casadi import MX, vertcat, Function, jacobian
import numpy as np
import numpy.testing as npt
from bioptim import CasadiFunctionInterface


class CasadiFunctionInterfaceTest(CasadiFunctionInterface):
    """
    This example implements a somewhat simple 5x1 function, with x and y inputs (x => 3x1; y => 4x1) of the form
        f(x, y) = np.array(
            [
                x[0] * y[1] + y[0] * y[0],
                x[1] * x[1] + 2 * y[1],
                x[0] * x[1] * x[2],
                x[2] * x[1] + 2 * y[3] * y[2],
                y[0] * y[1] * y[2] * y[3],
            ]
        )

    It implements the equation (5x1) and the jacobians for the inputs x (5x3) and y (5x4).
    """

    def __init__(self, opts={}):
        super(CasadiFunctionInterfaceTest, self).__init__("CasadiFunctionInterfaceTest", opts)

    def inputs_len(self) -> list[int]:
        return [3, 4]

    def outputs_len(self) -> list[int]:
        return [5]

    def function(self, *args):
        x, y = args
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

    def jacobians(self, *args):
        x, y = args
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
        return [jacobian_x, jacobian_y]


def test_penalty_minimize_time():
    """
    These tests seem to test the interface, but actually all the internal methods are also called, which is what should
    be tested.
    """

    # Computing the example
    interface_test = CasadiFunctionInterfaceTest()

    # Testing the interface
    npt.assert_equal(interface_test.inputs_len(), [3, 4])
    npt.assert_equal(interface_test.outputs_len(), [5])
    assert id(interface_test.mx_in()) == id(interface_test.mx_in())  # Calling twice returns the same object

    # Test the class can be called with DM
    x_num = np.array([1.1, 2.3, 3.5])
    y_num = np.array([4.2, 5.4, 6.6, 7.7])
    npt.assert_almost_equal(interface_test(x_num, y_num), np.array([[25.344, 16.09, 8.855, 154.77, 1152.5976]]).T)

    # Test the jacobian is correct
    x = MX.sym("x", interface_test.inputs_len()[0], 1)
    y = MX.sym("y", interface_test.inputs_len()[1], 1)
    jaco_x = Function("jaco_x", [x, y], [jacobian(interface_test(x, y), x)])
    jaco_y = Function("jaco_y", [x, y], [jacobian(interface_test(x, y), y)])

    # Computing the same equations (and derivative) by casadi
    real = vertcat(
        x[0] * y[1] + x[0] * y[0] * y[0],
        x[1] * x[1] + 2 * y[1],
        x[0] * x[1] * x[2],
        x[2] * x[1] * y[2] + 2 * y[3] * y[2],
        y[0] * y[1] * y[2] * y[3],
    )
    real_function = Function("real", [x, y], [real])
    jaco_x_real = Function("jaco_x_real", [x, y], [jacobian(real, x)])
    jaco_y_real = Function("jaco_y_real", [x, y], [jacobian(real, y)])

    npt.assert_almost_equal(np.array(interface_test(x_num, y_num)), real_function(x_num, y_num))
    npt.assert_almost_equal(np.array(jaco_x(x_num, y_num)), jaco_x_real(x_num, y_num))
    npt.assert_almost_equal(np.array(jaco_y(x_num, y_num)), jaco_y_real(x_num, y_num))
