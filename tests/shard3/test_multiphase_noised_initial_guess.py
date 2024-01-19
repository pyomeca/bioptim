# import pytest
# import os
# import numpy as np
# from bioptim import BiorbdModel, BoundsList, InitialGuessList, MagnitudeType, PhaseDynamics
#
#
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# def test_noisy_multiphase(phase_dynamics):
#     from bioptim.examples.getting_started import example_multiphase as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
#         phase_dynamics=phase_dynamics,
#         expand_dynamics=True,
#     )
#     bio_model = BiorbdModel(bioptim_folder + "/models/cube.bioMod")
#     n_shooting = [20, 30, 20]
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=0)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=0)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=1)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=1)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=2)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=2)
#
#     for bounds in x_bounds:
#         bounds["q"][1, [0, -1]] = 0
#         bounds["qdot"][:, [0, -1]] = 0
#     x_bounds[0]["q"][2, 0] = 0.0
#     x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]
#
#     x_init = InitialGuessList()
#     x_init.add("q", [1, 2, 1], phase=0)
#     x_init.add("qdot", [2, 1, 2], phase=0)
#     x_init.add("q", [1, 2, 1], phase=1)
#     x_init.add("qdot", [2, 1, 2], phase=1)
#     x_init.add("q", [1, 2, 1], phase=2)
#     x_init.add("qdot", [2, 1, 2], phase=2)
#
#     # Define control path constraint
#     u_bounds = BoundsList()
#     u_bounds.add("tau", min_bound=[-100] * bio_model.nb_tau, max_bound=[100] * bio_model.nb_tau, phase=0)
#     u_bounds.add("tau", min_bound=[-100] * bio_model.nb_tau, max_bound=[100] * bio_model.nb_tau, phase=1)
#     u_bounds.add("tau", min_bound=[-100] * bio_model.nb_tau, max_bound=[100] * bio_model.nb_tau, phase=2)
#
#     u_init = InitialGuessList()
#     u_init.add("tau", [1, 2, 1], phase=0)
#     u_init.add("tau", [1, 2, 1], phase=1)
#     u_init.add("tau", [1, 2, 1], phase=2)
#
#     x_init.add_noise(
#         bounds=x_bounds,
#         magnitude=[0.1, 0.1, 0.1],
#         n_shooting=[ns + 1 for ns in n_shooting],
#         bound_push=0.1,
#         seed=[42] * 3,
#         magnitude_type=MagnitudeType.RELATIVE,
#     )
#
#     u_init.add_noise(
#         bounds=u_bounds,
#         magnitude=0.1,
#         n_shooting=[ns for ns in n_shooting],
#         bound_push=0.1,
#         seed=[42] * 3,
#         magnitude_type=MagnitudeType.RELATIVE,
#     )
#
#     ocp.update_bounds(x_bounds, u_bounds)
#     ocp.update_initial_guess(x_init, u_init)
#     expected = [
#         [9.24724071e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [1.27042858e00],
#         [9.00000000e-01],
#         [1.51436719e00],
#         [-7.52858140e-01],
#         [-1.19681116e00],
#         [8.73838597e-01],
#         [1.13919637e00],
#         [9.00000000e-01],
#         [6.96873985e-01],
#         [2.53653480e00],
#         [3.88531633e00],
#         [5.21135032e00],
#         [1.05919509e00],
#         [9.00000000e-01],
#         [1.20423153e00],
#         [-2.51227927e00],
#         [2.72859817e00],
#         [-1.40802276e00],
#         [7.93611184e-01],
#         [9.00000000e-01],
#         [7.63389160e-01],
#         [5.79751926e00],
#         [5.86585883e00],
#         [-3.31582723e00],
#         [7.93596712e-01],
#         [9.00000000e-01],
#         [1.02521822e00],
#         [-3.34635429e00],
#         [6.50842452e-01],
#         [-6.42061164e-01],
#         [7.34850167e-01],
#         [9.00000000e-01],
#         [1.05869787e00],
#         [8.11840169e00],
#         [-3.78031969e00],
#         [-2.25721886e00],
#         [1.21970569e00],
#         [9.00000000e-01],
#         [6.03976429e-01],
#         [5.42112867e00],
#         [3.67971303e00],
#         [7.39973995e00],
#         [1.06066901e00],
#         [9.00000000e-01],
#         [1.59009745e00],
#         [-1.78605041e00],
#         [4.27712157e00],
#         [5.87195488e00],
#         [1.12484355e00],
#         [9.00000000e-01],
#         [1.34574210e00],
#         [-4.21379234e00],
#         [1.77003197e00],
#         [3.67640105e00],
#         [7.12350697e-01],
#         [9.00000000e-01],
#         [1.55229066e00],
#         [5.96420522e00],
#         [4.40507401e00],
#         [6.66791144e00],
#         [1.28194591e00],
#         [9.00000000e-01],
#         [1.49615468e00],
#         [4.59944605e00],
#         [9.22033164e-01],
#         [5.81605586e00],
#         [1.19946558e00],
#         [9.00000000e-01],
#         [1.12302474e00],
#         [4.87778895e00],
#         [1.28566916e00],
#         [-1.93867680e00],
#         [8.27403466e-01],
#         [9.00000000e-01],
#         [1.53014280e00],
#         [5.40888371e00],
#         [8.94535824e-02],
#         [6.93304186e00],
#         [8.09094980e-01],
#         [9.00000000e-01],
#         [4.82884427e-01],
#         [-3.35271277e00],
#         [-4.96375914e00],
#         [2.49438919e00],
#         [8.10042706e-01],
#         [9.00000000e-01],
#         [6.17960798e-01],
#         [2.21427890e-01],
#         [-3.92738165e00],
#         [5.86340693e00],
#         [8.82545346e-01],
#         [9.00000000e-01],
#         [4.28515757e-01],
#         [-2.82713176e00],
#         [-4.88823451e00],
#         [6.97743007e00],
#         [1.01485386e00],
#         [9.00000000e-01],
#         [7.80503620e-01],
#         [6.56289222e00],
#         [2.71418378e00],
#         [-2.87035784e-01],
#         [9.59167011e-01],
#         [9.00000000e-01],
#         [8.60107756e-01],
#         [3.54940996e00],
#         [-1.33287154e00],
#         [-2.90023204e00],
#         [8.74737484e-01],
#         [9.00000000e-01],
#         [7.12668719e-01],
#         [-1.24998091e-01],
#         [1.10770248e00],
#         [-1.41886758e00],
#         [1.06711174e00],
#         [-1.00000000e-01],
#         [1.41310374e00],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [9.24724071e-01],
#         [-1.00000000e-01],
#         [1.41310374e00],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [1.27042858e00],
#         [9.00000000e-01],
#         [8.19990921e-01],
#         [5.40507401e00],
#         [8.40094570e-02],
#         [4.16311908e00],
#         [1.13919637e00],
#         [9.00000000e-01],
#         [7.24714186e-01],
#         [1.92203316e00],
#         [4.99629141e00],
#         [5.28760905e00],
#         [1.05919509e00],
#         [9.00000000e-01],
#         [1.05365348e00],
#         [2.28566916e00],
#         [5.53307420e00],
#         [-1.29694386e00],
#         [7.93611184e-01],
#         [9.00000000e-01],
#         [5.48772073e-01],
#         [1.08945358e00],
#         [-5.19582226e00],
#         [4.86785122e00],
#         [7.93596712e-01],
#         [9.00000000e-01],
#         [1.37975193e00],
#         [-3.96375914e00],
#         [1.13505459e00],
#         [3.38513844e-01],
#         [7.34850167e-01],
#         [9.00000000e-01],
#         [4.65364571e-01],
#         [-2.92738165e00],
#         [-3.78439431e-02],
#         [3.66260410e00],
#         [1.21970569e00],
#         [9.00000000e-01],
#         [1.61184017e00],
#         [-3.88823451e00],
#         [-2.49209624e00],
#         [3.67798383e00],
#         [1.06066901e00],
#         [9.00000000e-01],
#         [1.34211287e00],
#         [3.71418378e00],
#         [-3.77691268e00],
#         [2.44955794e00],
#         [1.12484355e00],
#         [9.00000000e-01],
#         [6.21394959e-01],
#         [-3.32871544e-01],
#         [-1.04058794e00],
#         [-3.14857059e00],
#         [7.12350697e-01],
#         [9.00000000e-01],
#         [3.78620766e-01],
#         [2.10770248e00],
#         [6.56576749e00],
#         [6.21353543e00],
#         [1.28194591e00],
#         [9.00000000e-01],
#         [1.39642052e00],
#         [7.12163136e00],
#         [-1.22169748e00],
#         [-2.52144125e-01],
#         [1.19946558e00],
#         [9.00000000e-01],
#         [1.25994460e00],
#         [-1.15048676e00],
#         [1.23612992e00],
#         [-1.93932458e00],
#         [8.27403466e-01],
#         [9.00000000e-01],
#         [1.28777889e00],
#         [8.73838597e-01],
#         [3.55121148e00],
#         [-3.77078977e00],
#         [8.09094980e-01],
#         [9.00000000e-01],
#         [1.34088837e00],
#         [5.21135032e00],
#         [-7.13680957e-01],
#         [3.14219441e00],
#         [8.10042706e-01],
#         [9.00000000e-01],
#         [4.64728723e-01],
#         [-1.40802276e00],
#         [6.92858850e00],
#         [4.23133958e00],
#         [8.82545346e-01],
#         [9.00000000e-01],
#         [8.22142789e-01],
#         [-3.31582723e00],
#         [6.81128410e00],
#         [-4.07473650e00],
#         [1.01485386e00],
#         [9.00000000e-01],
#         [5.17286824e-01],
#         [-6.42061164e-01],
#         [-2.11919566e00],
#         [2.15196585e00],
#         [9.59167011e-01],
#         [9.00000000e-01],
#         [1.45628922e00],
#         [-2.25721886e00],
#         [9.65423705e-01],
#         [-1.43695545e00],
#         [8.74737484e-01],
#         [9.00000000e-01],
#         [1.15494100e00],
#         [7.39973995e00],
#         [-1.50223696e00],
#         [3.82429509e00],
#         [1.06711174e00],
#         [9.00000000e-01],
#         [7.87500191e-01],
#         [5.87195488e00],
#         [-1.70377409e00],
#         [-2.09203214e00],
#         [7.83696316e-01],
#         [9.00000000e-01],
#         [4.51551248e-01],
#         [3.67640105e00],
#         [-4.81965026e00],
#         [4.39939438e00],
#         [8.75286789e-01],
#         [9.00000000e-01],
#         [7.62473380e-01],
#         [6.66791144e00],
#         [2.37682603e00],
#         [5.76674384e-01],
#         [9.19817106e-01],
#         [9.00000000e-01],
#         [7.80318884e-01],
#         [5.81605586e00],
#         [1.03366560e00],
#         [7.48811090e00],
#         [9.73641991e-01],
#         [9.00000000e-01],
#         [1.28853163e00],
#         [-1.93867680e00],
#         [-4.63628424e00],
#         [-2.55504616e00],
#         [1.17110558e00],
#         [9.00000000e-01],
#         [1.17285982e00],
#         [6.93304186e00],
#         [-1.78161057e00],
#         [2.78086421e-03],
#         [8.19804269e-01],
#         [9.00000000e-01],
#         [1.48658588e00],
#         [2.49438919e00],
#         [6.13042043e00],
#         [-2.85723498e00],
#         [1.00854066e00],
#         [9.00000000e-01],
#         [9.65084245e-01],
#         [5.86340693e00],
#         [-2.27276180e00],
#         [7.33685740e00],
#         [1.05544874e00],
#         [9.00000000e-01],
#         [5.21968031e-01],
#         [6.97743007e00],
#         [-3.46238264e00],
#         [6.74178616e00],
#         [7.27870248e-01],
#         [9.00000000e-01],
#         [1.26797130e00],
#         [-2.87035784e-01],
#         [8.67459477e-01],
#         [-1.04179522e00],
#         [1.06452691e00],
#         [-1.00000000e-01],
#         [1.32771216e00],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [9.24724071e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [1.27042858e00],
#         [9.00000000e-01],
#         [1.51436719e00],
#         [-7.52858140e-01],
#         [-1.19681116e00],
#         [8.73838597e-01],
#         [1.13919637e00],
#         [9.00000000e-01],
#         [6.96873985e-01],
#         [2.53653480e00],
#         [3.88531633e00],
#         [5.21135032e00],
#         [1.05919509e00],
#         [9.00000000e-01],
#         [1.20423153e00],
#         [-2.51227927e00],
#         [2.72859817e00],
#         [-1.40802276e00],
#         [7.93611184e-01],
#         [9.00000000e-01],
#         [7.63389160e-01],
#         [5.79751926e00],
#         [5.86585883e00],
#         [-3.31582723e00],
#         [7.93596712e-01],
#         [9.00000000e-01],
#         [1.02521822e00],
#         [-3.34635429e00],
#         [6.50842452e-01],
#         [-6.42061164e-01],
#         [7.34850167e-01],
#         [9.00000000e-01],
#         [1.05869787e00],
#         [8.11840169e00],
#         [-3.78031969e00],
#         [-2.25721886e00],
#         [1.21970569e00],
#         [9.00000000e-01],
#         [6.03976429e-01],
#         [5.42112867e00],
#         [3.67971303e00],
#         [7.39973995e00],
#         [1.06066901e00],
#         [9.00000000e-01],
#         [1.59009745e00],
#         [-1.78605041e00],
#         [4.27712157e00],
#         [5.87195488e00],
#         [1.12484355e00],
#         [9.00000000e-01],
#         [1.34574210e00],
#         [-4.21379234e00],
#         [1.77003197e00],
#         [3.67640105e00],
#         [7.12350697e-01],
#         [9.00000000e-01],
#         [1.55229066e00],
#         [5.96420522e00],
#         [4.40507401e00],
#         [6.66791144e00],
#         [1.28194591e00],
#         [9.00000000e-01],
#         [1.49615468e00],
#         [4.59944605e00],
#         [9.22033164e-01],
#         [5.81605586e00],
#         [1.19946558e00],
#         [9.00000000e-01],
#         [1.12302474e00],
#         [4.87778895e00],
#         [1.28566916e00],
#         [-1.93867680e00],
#         [8.27403466e-01],
#         [9.00000000e-01],
#         [1.53014280e00],
#         [5.40888371e00],
#         [8.94535824e-02],
#         [6.93304186e00],
#         [8.09094980e-01],
#         [9.00000000e-01],
#         [4.82884427e-01],
#         [-3.35271277e00],
#         [-4.96375914e00],
#         [2.49438919e00],
#         [8.10042706e-01],
#         [9.00000000e-01],
#         [6.17960798e-01],
#         [2.21427890e-01],
#         [-3.92738165e00],
#         [5.86340693e00],
#         [8.82545346e-01],
#         [9.00000000e-01],
#         [4.28515757e-01],
#         [-2.82713176e00],
#         [-4.88823451e00],
#         [6.97743007e00],
#         [1.01485386e00],
#         [9.00000000e-01],
#         [7.80503620e-01],
#         [6.56289222e00],
#         [2.71418378e00],
#         [-2.87035784e-01],
#         [9.59167011e-01],
#         [9.00000000e-01],
#         [8.60107756e-01],
#         [3.54940996e00],
#         [-1.33287154e00],
#         [-2.90023204e00],
#         [8.74737484e-01],
#         [9.00000000e-01],
#         [7.12668719e-01],
#         [-1.24998091e-01],
#         [1.10770248e00],
#         [-1.41886758e00],
#         [1.06711174e00],
#         [-1.00000000e-01],
#         [1.67000000e00],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-1.00000000e-01],
#         [-4.01839525e00],
#         [6.47411579e00],
#         [-1.41184706e01],
#         [1.90285723e01],
#         [-1.24202456e01],
#         [8.07076404e-01],
#         [1.02797577e01],
#         [-6.31421406e00],
#         [-1.76244592e01],
#         [4.94633937e00],
#         [-3.34552627e00],
#         [1.73728161e01],
#         [-1.27592544e01],
#         [2.42799369e-01],
#         [-8.64880074e00],
#         [-1.27602192e01],
#         [1.34070385e01],
#         [7.50089137e00],
#         [-1.66766555e01],
#         [-1.00130487e01],
#         [-6.53155696e00],
#         [1.56470458e01],
#         [2.56937754e00],
#         [1.80272085e00],
#         [5.04460047e00],
#         [5.69658275e00],
#         [2.86841117e00],
#         [9.32290311e00],
#         [-1.61419835e01],
#         [-1.16058218e01],
#         [-1.81766202e01],
#         [6.30179408e00],
#         [1.97833851e01],
#         [1.97963941e01],
#         [-1.11790351e01],
#         [1.20053129e01],
#         [1.42977056e01],
#         [-1.53979363e01],
#         [1.85799577e01],
#         [-1.05064356e01],
#         [1.99554215e01],
#         [1.67930940e01],
#         [-1.17270013e01],
#         [2.06252813e01],
#         [4.91599915e00],
#         [-1.16638196e01],
#         [1.43358939e01],
#         [1.78749694e01],
#         [-6.83031028e00],
#         [-5.81544923e00],
#         [-1.54602999e01],
#         [1.99025727e00],
#         [-1.40931154e01],
#         [-1.11606855e01],
#         [-1.72219925e00],
#         [9.36932106e00],
#         [-1.71909084e01],
#         [-7.35083439e00],
#         [-3.93900250e-01],
#         [-5.98678677e00],
#         [-4.01839525e00],
#         [6.30179408e00],
#         [-3.45290841e00],
#         [1.90285723e01],
#         [-1.11790351e01],
#         [-8.14603873e00],
#         [1.02797577e01],
#         [-1.53979363e01],
#         [1.41495004e01],
#         [4.94633937e00],
#         [1.99554215e01],
#         [-4.72986693e00],
#         [-1.27592544e01],
#         [2.06252813e01],
#         [-7.76261961e00],
#         [-1.27602192e01],
#         [1.43358939e01],
#         [2.70784333e00],
#         [-1.66766555e01],
#         [-5.81544923e00],
#         [-1.33630310e01],
#         [1.56470458e01],
#         [-1.40931154e01],
#         [1.30878792e01],
#         [5.04460047e00],
#         [9.36932106e00],
#         [-1.60179743e01],
#         [9.32290311e00],
#         [-3.93900250e-01],
#         [2.04754775e01],
#         [-1.81766202e01],
#         [-1.31184706e01],
#         [1.18897908e01],
#         [1.97963941e01],
#         [1.80707640e00],
#         [-1.10513727e01],
#         [1.42977056e01],
#         [-1.66244592e01],
#         [-1.87791153e01],
#         [-1.05064356e01],
#         [1.83728161e01],
#         [1.36184571e01],
#         [-1.17270013e01],
#         [-7.64880074e00],
#         [9.27429375e00],
#         [-1.16638196e01],
#         [8.50089137e00],
#         [1.01602867e01],
#         [-6.83031028e00],
#         [-5.53155696e00],
#         [1.18508139e01],
#         [1.99025727e00],
#         [2.80272085e00],
#         [-1.60382139e01],
#         [-1.72219925e00],
#         [3.86841117e00],
#         [-4.66137086e00],
#         [-7.35083439e00],
#         [-1.06058218e01],
#         [-1.43652376e01],
#         [5.47411579e00],
#         [2.07833851e01],
#         [1.55241370e01],
#         [-1.34202456e01],
#         [1.30053129e01],
#         [5.93192507e00],
#         [-7.31421406e00],
#         [1.95799577e01],
#         [-5.76407901e00],
#         [-4.34552627e00],
#         [1.77930940e01],
#         [-1.64576660e01],
#         [-7.57200631e-01],
#         [5.91599915e00],
#         [-6.56070713e00],
#         [1.24070385e01],
#         [1.88749694e01],
#         [-5.99266712e00],
#         [-1.10130487e01],
#         [-1.44602999e01],
#         [1.01842471e01],
#         [1.56937754e00],
#         [-1.01606855e01],
#         [6.50229885e00],
#         [4.69658275e00],
#         [-1.61909084e01],
#         [1.64885097e01],
#         [-1.71419835e01],
#         [-4.98678677e00],
#         [-1.11402994e-01],
#         [-4.01839525e00],
#         [6.47411579e00],
#         [-1.41184706e01],
#         [1.90285723e01],
#         [-1.24202456e01],
#         [8.07076404e-01],
#         [1.02797577e01],
#         [-6.31421406e00],
#         [-1.76244592e01],
#         [4.94633937e00],
#         [-3.34552627e00],
#         [1.73728161e01],
#         [-1.27592544e01],
#         [2.42799369e-01],
#         [-8.64880074e00],
#         [-1.27602192e01],
#         [1.34070385e01],
#         [7.50089137e00],
#         [-1.66766555e01],
#         [-1.00130487e01],
#         [-6.53155696e00],
#         [1.56470458e01],
#         [2.56937754e00],
#         [1.80272085e00],
#         [5.04460047e00],
#         [5.69658275e00],
#         [2.86841117e00],
#         [9.32290311e00],
#         [-1.61419835e01],
#         [-1.16058218e01],
#         [-1.81766202e01],
#         [6.30179408e00],
#         [1.97833851e01],
#         [1.97963941e01],
#         [-1.11790351e01],
#         [1.20053129e01],
#         [1.42977056e01],
#         [-1.53979363e01],
#         [1.85799577e01],
#         [-1.05064356e01],
#         [1.99554215e01],
#         [1.67930940e01],
#         [-1.17270013e01],
#         [2.06252813e01],
#         [4.91599915e00],
#         [-1.16638196e01],
#         [1.43358939e01],
#         [1.78749694e01],
#         [-6.83031028e00],
#         [-5.81544923e00],
#         [-1.54602999e01],
#         [1.99025727e00],
#         [-1.40931154e01],
#         [-1.11606855e01],
#         [-1.72219925e00],
#         [9.36932106e00],
#         [-1.71909084e01],
#         [-7.35083439e00],
#         [-3.93900250e-01],
#         [-5.98678677e00],
#     ]
#
#     np.testing.assert_almost_equal(ocp.init_vector, expected)
#
#
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize(
#     "magnitude, raised_str",
#     [
#         (None, "'magnitude' must be specified to generate noised initial guess"),
#         (tuple(np.ones(1) * 0.1), "'magnitude' must be an instance of int, float, list, or ndarray"),
#         ({"q": [0.1, 0.1]}, "Magnitude of all the elements must be specified, but qdot is missing"),
#     ],
# )
# def test_add_wrong_magnitude(magnitude, raised_str, phase_dynamics):
#     from bioptim.examples.getting_started import example_multiphase as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
#         phase_dynamics=phase_dynamics,
#         expand_dynamics=True,
#     )
#     bio_model = BiorbdModel(bioptim_folder + "/models/cube.bioMod")
#     n_shooting = [20, 30, 20]
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=0)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=0)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=1)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=1)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=2)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=2)
#
#     x_init = InitialGuessList()
#     x_init.add("q", [1, 2, 1], phase=0)
#     x_init.add("qdot", [2, 1, 2], phase=0)
#     x_init.add("q", [1, 2, 1], phase=1)
#     x_init.add("qdot", [2, 1, 2], phase=1)
#     x_init.add("q", [1, 2, 1], phase=2)
#     x_init.add("qdot", [2, 1, 2], phase=2)
#
#     with pytest.raises(ValueError, match=raised_str):
#         x_init.add_noise(
#             bounds=x_bounds,
#             magnitude=magnitude,  # n phase
#             n_shooting=[ns for ns in n_shooting],
#             bound_push=np.array((0.1)),
#             seed=42,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
#
#
# @pytest.mark.parametrize(
#     "bound_push, raised_str",
#     [
#         (None, "'bound_push' must be specified to generate noised initial guess"),
#         (tuple(np.ones(1) * 0.1), "'bound_push' must be an instance of int, float, list or ndarray"),
#         ([0.1, 0.1], f"Invalid size of 'bound_push', 'bound_push' as list must be size 1 or 3"),
#         (np.ones((2, 2)) * 0.1, "'bound_push' must be a 1 dimension array'"),
#         (np.ones(2) * 0.1, f"Invalid size of 'bound_push', 'bound_push' as array must be size 1 or 3"),
#     ],
# )
# def test_add_wrong_bound_push(bound_push, raised_str):
#     from bioptim.examples.getting_started import example_multiphase as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
#         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
#         expand_dynamics=True,
#     )
#     bio_model = BiorbdModel(bioptim_folder + "/models/cube.bioMod")
#     n_shooting = [20, 30, 20]
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=0)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=0)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=1)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=1)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=2)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=2)
#
#     x_init = InitialGuessList()
#     x_init.add("q", [1, 2, 1], phase=0)
#     x_init.add("qdot", [2, 1, 2], phase=0)
#     x_init.add("q", [1, 2, 1], phase=1)
#     x_init.add("qdot", [2, 1, 2], phase=1)
#     x_init.add("q", [1, 2, 1], phase=2)
#     x_init.add("qdot", [2, 1, 2], phase=2)
#
#     with pytest.raises(ValueError, match=raised_str):
#         x_init.add_noise(
#             bounds=x_bounds,
#             magnitude=0.1,
#             n_shooting=[ns + 1 for ns in n_shooting],
#             bound_push=bound_push,
#             seed=42,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
#
#
# @pytest.mark.parametrize(
#     "seed, raised_str",
#     [
#         (0.1, "Seed must be an integer, dict or a list of these"),
#         ([0.1, 0.1], f"Invalid size of 'seed', 'seed' as list must be size 1 or 3"),
#     ],
# )
# def test_add_wrong_seed(seed, raised_str):
#     from bioptim.examples.getting_started import example_multiphase as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
#         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
#         expand_dynamics=True,
#     )
#     bio_model = BiorbdModel(bioptim_folder + "/models/cube.bioMod")
#     n_shooting = [20, 30, 20]
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=0)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=0)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=1)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=1)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=2)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=2)
#
#     x_init = InitialGuessList()
#     x_init.add("q", [1, 2, 1], phase=0)
#     x_init.add("qdot", [2, 1, 2], phase=0)
#     x_init.add("q", [1, 2, 1], phase=1)
#     x_init.add("qdot", [2, 1, 2], phase=1)
#     x_init.add("q", [1, 2, 1], phase=2)
#     x_init.add("qdot", [2, 1, 2], phase=2)
#
#     with pytest.raises(ValueError, match=raised_str):
#         x_init.add_noise(
#             bounds=x_bounds,
#             magnitude=0.1,
#             n_shooting=[ns + 1 for ns in n_shooting],
#             bound_push=0.1,
#             seed=seed,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
#
#
# def test_add_wrong_bounds():
#     from bioptim.examples.getting_started import example_multiphase as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
#         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
#         expand_dynamics=True,
#     )
#     bio_model = BiorbdModel(bioptim_folder + "/models/cube.bioMod")
#     n_shooting = [20, 30, 20]
#
#     nb_phases = ocp.n_phases
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=0)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=0)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=1)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=1)
#     # Missing phase=2
#
#     x_init = InitialGuessList()
#     x_init.add("q", [1, 2, 1], phase=0)
#     x_init.add("qdot", [2, 1, 2], phase=0)
#     x_init.add("q", [1, 2, 1], phase=1)
#     x_init.add("qdot", [2, 1, 2], phase=1)
#     x_init.add("q", [1, 2, 1], phase=2)
#     x_init.add("qdot", [2, 1, 2], phase=2)
#
#     with pytest.raises(ValueError, match="bounds must be specified to generate noised initial guess"):
#         x_init.add_noise(
#             bounds=None,
#             magnitude=0.1,
#             n_shooting=[ns + 1 for ns in n_shooting],
#             bound_push=0.1,
#             seed=42,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
#     with pytest.raises(ValueError, match=f"Invalid size of 'bounds', 'bounds' must be size {nb_phases}"):
#         x_init.add_noise(
#             bounds=x_bounds,
#             magnitude=0.1,
#             n_shooting=[ns + 1 for ns in n_shooting],
#             bound_push=0.1,
#             seed=42,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
#
#
# def test_add_wrong_n_shooting():
#     from bioptim.examples.getting_started import example_multiphase as ocp_module
#
#     bioptim_folder = os.path.dirname(ocp_module.__file__)
#     ocp = ocp_module.prepare_ocp(
#         biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
#         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
#         expand_dynamics=True,
#     )
#     bio_model = BiorbdModel(bioptim_folder + "/models/cube.bioMod")
#
#     nb_phases = ocp.n_phases
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=0)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=0)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=1)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=1)
#     x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"), phase=2)
#     x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"), phase=2)
#
#     x_init = InitialGuessList()
#     x_init.add("q", [1, 2, 1], phase=0)
#     x_init.add("qdot", [2, 1, 2], phase=0)
#     x_init.add("q", [1, 2, 1], phase=1)
#     x_init.add("qdot", [2, 1, 2], phase=1)
#     x_init.add("q", [1, 2, 1], phase=2)
#     x_init.add("qdot", [2, 1, 2], phase=2)
#
#     with pytest.raises(ValueError, match="n_shooting must be specified to generate noised initial guess"):
#         x_init.add_noise(
#             bounds=x_bounds,
#             magnitude=0.1,
#             bound_push=0.1,
#             seed=42,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
#     with pytest.raises(ValueError, match=f"Invalid size of 'n_shooting', 'n_shooting' must be len {nb_phases}"):
#         x_init.add_noise(
#             bounds=x_bounds,
#             n_shooting=[20, 30],
#             magnitude=0.1,
#             bound_push=0.1,
#             seed=42,
#             magnitude_type=MagnitudeType.RELATIVE,
#         )
