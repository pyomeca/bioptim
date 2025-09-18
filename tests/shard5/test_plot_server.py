from bioptim.gui.online_callback_server import _serialize_xydata, _deserialize_xydata
from bioptim.gui.plot import PlotOcp
from bioptim.gui.online_callback_server import _ResponseHeader
from bioptim.optimization.optimization_vector import OptimizationVectorHelper
from casadi import DM
import numpy as np

from ..utils import TestUtils


def test_serialize_deserialize():
    # Prepare a set of data to serialize and deserialize
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=40,
    )

    dummy_phase_times = OptimizationVectorHelper.extract_step_times(ocp, DM(np.ones(ocp.n_phases)))
    plotter = PlotOcp(ocp, dummy_phase_times=dummy_phase_times, show_bounds=True, only_initialize_variables=True)

    np.random.seed(42)
    xdata, ydata = plotter.parse_data(**{"x": np.random.rand(ocp.variables_vector.shape[0])[:, None]})

    # Serialize and deserialize the data
    serialized_data = _serialize_xydata(xdata, ydata)
    deserialized_xdata, deserialized_ydata = _deserialize_xydata(serialized_data)

    # Compare the outputs
    for x_phase, deserialized_x_phase in zip(xdata, deserialized_xdata):
        for x_node, deserialized_x_node in zip(x_phase, deserialized_x_phase):
            assert np.allclose(x_node, DM(deserialized_x_node))

    for y_variable, deserialized_y_variable in zip(ydata, deserialized_ydata):
        if isinstance(y_variable, np.ndarray):
            assert np.allclose(y_variable, deserialized_y_variable[0], equal_nan=True)
        else:
            for y_phase, deserialized_y_phase in zip(y_variable, deserialized_y_variable):
                assert np.allclose(y_phase, deserialized_y_phase)


def test_response_header():
    # Make sure all the response have the same length
    response_len = _ResponseHeader.response_len()
    for response in _ResponseHeader:
        assert len(response) == response_len
        # Make sure encoding provides a constant length
        assert len(response.encode()) == response_len

    # Make sure equality works
    assert _ResponseHeader.OK == _ResponseHeader.OK
    assert _ResponseHeader.OK.value == _ResponseHeader.OK
    assert _ResponseHeader.OK.encode().decode() == _ResponseHeader.OK
    assert _ResponseHeader.OK == _ResponseHeader.OK.encode().decode()
    assert not (_ResponseHeader.OK != _ResponseHeader.OK)
    assert not (_ResponseHeader.OK.encode().decode() != _ResponseHeader.OK)
    assert not (_ResponseHeader.OK != _ResponseHeader.OK.encode().decode())
    assert not (_ResponseHeader.OK.value == _ResponseHeader.OK.encode().decode())
    assert _ResponseHeader.OK != _ResponseHeader.NOK
    assert _ResponseHeader.NOK == _ResponseHeader.NOK
