import os
from bioptim.limits.path_conditions import Bounds


def test_pendulum_show_bounds():
    """
    It doesn"t test that bounds are shown, b
    but we test that the bounds are added to the plot"
    """
    from bioptim.examples.sqp_method import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
        assume_phase_dynamics=True,
        expand_dynamics=True,
    )
    # Test the keys in the dict ocp.nlp[0].plot
    assert ["q_states", "qdot_states", "tau_controls"] == list(ocp.nlp[0].plot.keys())

    for key, plot in ocp.nlp[0].plot.items():
        assert isinstance(plot.bounds, Bounds)
        assert plot.bounds.shape == (2, 3)
        if "states" in key:
            state = key.split("_")[0]
            assert ocp.nlp[0].x_bounds[state] == plot.bounds
        elif "controls" in key:
            control = key.split("_")[0]
            assert ocp.nlp[0].u_bounds[control] == plot.bounds
