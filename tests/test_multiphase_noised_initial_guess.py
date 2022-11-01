import pytest
import os
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX
from bioptim.misc.enums import MagnitudeType
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    ParameterList,
    InterpolationType,
    InitialGuess,
    InitialGuessList,
    NoisedInitialGuess,
    Objective,
    ObjectiveFcn,
    OdeSolver,
)

from .utils import TestUtils


def test_noisy_multiphase():
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
    )
    biorbd_model = biorbd.Model(bioptim_folder + "/models/cube.bioMod")
    n_shooting = [20, 30, 20]

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    x_init = InitialGuessList()
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])

    u_bounds = BoundsList()
    u_bounds.add([-100] * biorbd_model.nbGeneralizedTorque(), [100] * biorbd_model.nbGeneralizedTorque())
    u_bounds.add([-100] * biorbd_model.nbGeneralizedTorque(), [100] * biorbd_model.nbGeneralizedTorque())
    u_bounds.add([-100] * biorbd_model.nbGeneralizedTorque(), [100] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([1, 2, 1])
    u_init.add([1, 2, 1])
    u_init.add([1, 2, 1])

    x_init.add_noise(
        bounds=x_bounds,
        magnitude=0.1,  # n phase
        n_shooting=[ns for ns in n_shooting],
        bound_push=0.1,
        seed=42,
        magnitude_type=MagnitudeType.RELATIVE,
    )

    u_init.add_noise(
        bounds=u_bounds,
        magnitude=0.1,  # n phase
        n_shooting=[ns - 1 for ns in n_shooting],
        bound_push=0.1,
        seed=42,
        magnitude_type=MagnitudeType.RELATIVE,
    )

    # ocp.isdef_x_init = False
    # ocp.isdef_u_init = False
    # ocp.isdef_x_bounds = False
    # ocp.isdef_u_bounds = False

    ocp.update_bounds(x_bounds, u_bounds)
    ocp.update_initial_guess(x_init, u_init)
    expected = [
        [9.24724071e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [1.27042858e00],
        [9.00000000e-01],
        [1.51436719e00],
        [-7.52858140e-01],
        [-1.19681116e00],
        [8.73838597e-01],
        [1.13919637e00],
        [9.00000000e-01],
        [6.96873985e-01],
        [2.53653480e00],
        [3.88531633e00],
        [5.21135032e00],
        [1.05919509e00],
        [9.00000000e-01],
        [1.20423153e00],
        [-2.51227927e00],
        [2.72859817e00],
        [-1.40802276e00],
        [7.93611184e-01],
        [9.00000000e-01],
        [7.63389160e-01],
        [5.79751926e00],
        [5.86585883e00],
        [-3.31582723e00],
        [7.93596712e-01],
        [9.00000000e-01],
        [1.02521822e00],
        [-3.34635429e00],
        [6.50842452e-01],
        [-6.42061164e-01],
        [7.34850167e-01],
        [9.00000000e-01],
        [1.05869787e00],
        [8.11840169e00],
        [-3.78031969e00],
        [-2.25721886e00],
        [1.21970569e00],
        [9.00000000e-01],
        [6.03976429e-01],
        [5.42112867e00],
        [3.67971303e00],
        [7.39973995e00],
        [1.06066901e00],
        [9.00000000e-01],
        [1.59009745e00],
        [-1.78605041e00],
        [4.27712157e00],
        [5.87195488e00],
        [1.12484355e00],
        [9.00000000e-01],
        [1.34574210e00],
        [-4.21379234e00],
        [1.77003197e00],
        [3.67640105e00],
        [7.12350697e-01],
        [9.00000000e-01],
        [1.55229066e00],
        [5.96420522e00],
        [4.40507401e00],
        [6.66791144e00],
        [1.28194591e00],
        [9.00000000e-01],
        [1.49615468e00],
        [4.59944605e00],
        [9.22033164e-01],
        [5.81605586e00],
        [1.19946558e00],
        [9.00000000e-01],
        [1.12302474e00],
        [4.87778895e00],
        [1.28566916e00],
        [-1.93867680e00],
        [8.27403466e-01],
        [9.00000000e-01],
        [1.53014280e00],
        [5.40888371e00],
        [8.94535824e-02],
        [6.93304186e00],
        [8.09094980e-01],
        [9.00000000e-01],
        [4.82884427e-01],
        [-3.35271277e00],
        [-4.96375914e00],
        [2.49438919e00],
        [8.10042706e-01],
        [9.00000000e-01],
        [6.17960798e-01],
        [2.21427890e-01],
        [-3.92738165e00],
        [5.86340693e00],
        [8.82545346e-01],
        [9.00000000e-01],
        [4.28515757e-01],
        [-2.82713176e00],
        [-4.88823451e00],
        [6.97743007e00],
        [1.01485386e00],
        [9.00000000e-01],
        [7.80503620e-01],
        [6.56289222e00],
        [2.71418378e00],
        [-2.87035784e-01],
        [9.59167011e-01],
        [9.00000000e-01],
        [8.60107756e-01],
        [3.54940996e00],
        [-1.33287154e00],
        [-2.90023204e00],
        [8.74737484e-01],
        [9.00000000e-01],
        [7.12668719e-01],
        [-1.24998091e-01],
        [1.10770248e00],
        [-1.41886758e00],
        [1.06711174e00],
        [-1.00000000e-01],
        [1.41310374e00],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [9.24724071e-01],
        [-1.00000000e-01],
        [1.41310374e00],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [1.27042858e00],
        [9.00000000e-01],
        [8.19990921e-01],
        [5.40507401e00],
        [8.40094570e-02],
        [4.16311908e00],
        [1.13919637e00],
        [9.00000000e-01],
        [7.24714186e-01],
        [1.92203316e00],
        [4.99629141e00],
        [5.28760905e00],
        [1.05919509e00],
        [9.00000000e-01],
        [1.05365348e00],
        [2.28566916e00],
        [5.53307420e00],
        [-1.29694386e00],
        [7.93611184e-01],
        [9.00000000e-01],
        [5.48772073e-01],
        [1.08945358e00],
        [-5.19582226e00],
        [4.86785122e00],
        [7.93596712e-01],
        [9.00000000e-01],
        [1.37975193e00],
        [-3.96375914e00],
        [1.13505459e00],
        [3.38513844e-01],
        [7.34850167e-01],
        [9.00000000e-01],
        [4.65364571e-01],
        [-2.92738165e00],
        [-3.78439431e-02],
        [3.66260410e00],
        [1.21970569e00],
        [9.00000000e-01],
        [1.61184017e00],
        [-3.88823451e00],
        [-2.49209624e00],
        [3.67798383e00],
        [1.06066901e00],
        [9.00000000e-01],
        [1.34211287e00],
        [3.71418378e00],
        [-3.77691268e00],
        [2.44955794e00],
        [1.12484355e00],
        [9.00000000e-01],
        [6.21394959e-01],
        [-3.32871544e-01],
        [-1.04058794e00],
        [-3.14857059e00],
        [7.12350697e-01],
        [9.00000000e-01],
        [3.78620766e-01],
        [2.10770248e00],
        [6.56576749e00],
        [6.21353543e00],
        [1.28194591e00],
        [9.00000000e-01],
        [1.39642052e00],
        [7.12163136e00],
        [-1.22169748e00],
        [-2.52144125e-01],
        [1.19946558e00],
        [9.00000000e-01],
        [1.25994460e00],
        [-1.15048676e00],
        [1.23612992e00],
        [-1.93932458e00],
        [8.27403466e-01],
        [9.00000000e-01],
        [1.28777889e00],
        [8.73838597e-01],
        [3.55121148e00],
        [-3.77078977e00],
        [8.09094980e-01],
        [9.00000000e-01],
        [1.34088837e00],
        [5.21135032e00],
        [-7.13680957e-01],
        [3.14219441e00],
        [8.10042706e-01],
        [9.00000000e-01],
        [4.64728723e-01],
        [-1.40802276e00],
        [6.92858850e00],
        [4.23133958e00],
        [8.82545346e-01],
        [9.00000000e-01],
        [8.22142789e-01],
        [-3.31582723e00],
        [6.81128410e00],
        [-4.07473650e00],
        [1.01485386e00],
        [9.00000000e-01],
        [5.17286824e-01],
        [-6.42061164e-01],
        [-2.11919566e00],
        [2.15196585e00],
        [9.59167011e-01],
        [9.00000000e-01],
        [1.45628922e00],
        [-2.25721886e00],
        [9.65423705e-01],
        [-1.43695545e00],
        [8.74737484e-01],
        [9.00000000e-01],
        [1.15494100e00],
        [7.39973995e00],
        [-1.50223696e00],
        [3.82429509e00],
        [1.06711174e00],
        [9.00000000e-01],
        [7.87500191e-01],
        [5.87195488e00],
        [-1.70377409e00],
        [-2.09203214e00],
        [7.83696316e-01],
        [9.00000000e-01],
        [4.51551248e-01],
        [3.67640105e00],
        [-4.81965026e00],
        [4.39939438e00],
        [8.75286789e-01],
        [9.00000000e-01],
        [7.62473380e-01],
        [6.66791144e00],
        [2.37682603e00],
        [5.76674384e-01],
        [9.19817106e-01],
        [9.00000000e-01],
        [7.80318884e-01],
        [5.81605586e00],
        [1.03366560e00],
        [7.48811090e00],
        [9.73641991e-01],
        [9.00000000e-01],
        [1.28853163e00],
        [-1.93867680e00],
        [-4.63628424e00],
        [-2.55504616e00],
        [1.17110558e00],
        [9.00000000e-01],
        [1.17285982e00],
        [6.93304186e00],
        [-1.78161057e00],
        [2.78086421e-03],
        [8.19804269e-01],
        [9.00000000e-01],
        [1.48658588e00],
        [2.49438919e00],
        [6.13042043e00],
        [-2.85723498e00],
        [1.00854066e00],
        [9.00000000e-01],
        [9.65084245e-01],
        [5.86340693e00],
        [-2.27276180e00],
        [7.33685740e00],
        [1.05544874e00],
        [9.00000000e-01],
        [5.21968031e-01],
        [6.97743007e00],
        [-3.46238264e00],
        [6.74178616e00],
        [7.27870248e-01],
        [9.00000000e-01],
        [1.26797130e00],
        [-2.87035784e-01],
        [8.67459477e-01],
        [-1.04179522e00],
        [1.06452691e00],
        [-1.00000000e-01],
        [1.32771216e00],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [9.24724071e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [1.27042858e00],
        [9.00000000e-01],
        [1.51436719e00],
        [-7.52858140e-01],
        [-1.19681116e00],
        [8.73838597e-01],
        [1.13919637e00],
        [9.00000000e-01],
        [6.96873985e-01],
        [2.53653480e00],
        [3.88531633e00],
        [5.21135032e00],
        [1.05919509e00],
        [9.00000000e-01],
        [1.20423153e00],
        [-2.51227927e00],
        [2.72859817e00],
        [-1.40802276e00],
        [7.93611184e-01],
        [9.00000000e-01],
        [7.63389160e-01],
        [5.79751926e00],
        [5.86585883e00],
        [-3.31582723e00],
        [7.93596712e-01],
        [9.00000000e-01],
        [1.02521822e00],
        [-3.34635429e00],
        [6.50842452e-01],
        [-6.42061164e-01],
        [7.34850167e-01],
        [9.00000000e-01],
        [1.05869787e00],
        [8.11840169e00],
        [-3.78031969e00],
        [-2.25721886e00],
        [1.21970569e00],
        [9.00000000e-01],
        [6.03976429e-01],
        [5.42112867e00],
        [3.67971303e00],
        [7.39973995e00],
        [1.06066901e00],
        [9.00000000e-01],
        [1.59009745e00],
        [-1.78605041e00],
        [4.27712157e00],
        [5.87195488e00],
        [1.12484355e00],
        [9.00000000e-01],
        [1.34574210e00],
        [-4.21379234e00],
        [1.77003197e00],
        [3.67640105e00],
        [7.12350697e-01],
        [9.00000000e-01],
        [1.55229066e00],
        [5.96420522e00],
        [4.40507401e00],
        [6.66791144e00],
        [1.28194591e00],
        [9.00000000e-01],
        [1.49615468e00],
        [4.59944605e00],
        [9.22033164e-01],
        [5.81605586e00],
        [1.19946558e00],
        [9.00000000e-01],
        [1.12302474e00],
        [4.87778895e00],
        [1.28566916e00],
        [-1.93867680e00],
        [8.27403466e-01],
        [9.00000000e-01],
        [1.53014280e00],
        [5.40888371e00],
        [8.94535824e-02],
        [6.93304186e00],
        [8.09094980e-01],
        [9.00000000e-01],
        [4.82884427e-01],
        [-3.35271277e00],
        [-4.96375914e00],
        [2.49438919e00],
        [8.10042706e-01],
        [9.00000000e-01],
        [6.17960798e-01],
        [2.21427890e-01],
        [-3.92738165e00],
        [5.86340693e00],
        [8.82545346e-01],
        [9.00000000e-01],
        [4.28515757e-01],
        [-2.82713176e00],
        [-4.88823451e00],
        [6.97743007e00],
        [1.01485386e00],
        [9.00000000e-01],
        [7.80503620e-01],
        [6.56289222e00],
        [2.71418378e00],
        [-2.87035784e-01],
        [9.59167011e-01],
        [9.00000000e-01],
        [8.60107756e-01],
        [3.54940996e00],
        [-1.33287154e00],
        [-2.90023204e00],
        [8.74737484e-01],
        [9.00000000e-01],
        [7.12668719e-01],
        [-1.24998091e-01],
        [1.10770248e00],
        [-1.41886758e00],
        [1.06711174e00],
        [-1.00000000e-01],
        [1.67000000e00],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-1.00000000e-01],
        [-4.01839525e00],
        [6.47411579e00],
        [-1.41184706e01],
        [1.90285723e01],
        [-1.24202456e01],
        [8.07076404e-01],
        [1.02797577e01],
        [-6.31421406e00],
        [-1.76244592e01],
        [4.94633937e00],
        [-3.34552627e00],
        [1.73728161e01],
        [-1.27592544e01],
        [2.42799369e-01],
        [-8.64880074e00],
        [-1.27602192e01],
        [1.34070385e01],
        [7.50089137e00],
        [-1.66766555e01],
        [-1.00130487e01],
        [-6.53155696e00],
        [1.56470458e01],
        [2.56937754e00],
        [1.80272085e00],
        [5.04460047e00],
        [5.69658275e00],
        [2.86841117e00],
        [9.32290311e00],
        [-1.61419835e01],
        [-1.16058218e01],
        [-1.81766202e01],
        [6.30179408e00],
        [1.97833851e01],
        [1.97963941e01],
        [-1.11790351e01],
        [1.20053129e01],
        [1.42977056e01],
        [-1.53979363e01],
        [1.85799577e01],
        [-1.05064356e01],
        [1.99554215e01],
        [1.67930940e01],
        [-1.17270013e01],
        [2.06252813e01],
        [4.91599915e00],
        [-1.16638196e01],
        [1.43358939e01],
        [1.78749694e01],
        [-6.83031028e00],
        [-5.81544923e00],
        [-1.54602999e01],
        [1.99025727e00],
        [-1.40931154e01],
        [-1.11606855e01],
        [-1.72219925e00],
        [9.36932106e00],
        [-1.71909084e01],
        [-7.35083439e00],
        [-3.93900250e-01],
        [-5.98678677e00],
        [-4.01839525e00],
        [6.30179408e00],
        [-3.45290841e00],
        [1.90285723e01],
        [-1.11790351e01],
        [-8.14603873e00],
        [1.02797577e01],
        [-1.53979363e01],
        [1.41495004e01],
        [4.94633937e00],
        [1.99554215e01],
        [-4.72986693e00],
        [-1.27592544e01],
        [2.06252813e01],
        [-7.76261961e00],
        [-1.27602192e01],
        [1.43358939e01],
        [2.70784333e00],
        [-1.66766555e01],
        [-5.81544923e00],
        [-1.33630310e01],
        [1.56470458e01],
        [-1.40931154e01],
        [1.30878792e01],
        [5.04460047e00],
        [9.36932106e00],
        [-1.60179743e01],
        [9.32290311e00],
        [-3.93900250e-01],
        [2.04754775e01],
        [-1.81766202e01],
        [-1.31184706e01],
        [1.18897908e01],
        [1.97963941e01],
        [1.80707640e00],
        [-1.10513727e01],
        [1.42977056e01],
        [-1.66244592e01],
        [-1.87791153e01],
        [-1.05064356e01],
        [1.83728161e01],
        [1.36184571e01],
        [-1.17270013e01],
        [-7.64880074e00],
        [9.27429375e00],
        [-1.16638196e01],
        [8.50089137e00],
        [1.01602867e01],
        [-6.83031028e00],
        [-5.53155696e00],
        [1.18508139e01],
        [1.99025727e00],
        [2.80272085e00],
        [-1.60382139e01],
        [-1.72219925e00],
        [3.86841117e00],
        [-4.66137086e00],
        [-7.35083439e00],
        [-1.06058218e01],
        [-1.43652376e01],
        [5.47411579e00],
        [2.07833851e01],
        [1.55241370e01],
        [-1.34202456e01],
        [1.30053129e01],
        [5.93192507e00],
        [-7.31421406e00],
        [1.95799577e01],
        [-5.76407901e00],
        [-4.34552627e00],
        [1.77930940e01],
        [-1.64576660e01],
        [-7.57200631e-01],
        [5.91599915e00],
        [-6.56070713e00],
        [1.24070385e01],
        [1.88749694e01],
        [-5.99266712e00],
        [-1.10130487e01],
        [-1.44602999e01],
        [1.01842471e01],
        [1.56937754e00],
        [-1.01606855e01],
        [6.50229885e00],
        [4.69658275e00],
        [-1.61909084e01],
        [1.64885097e01],
        [-1.71419835e01],
        [-4.98678677e00],
        [-1.11402994e-01],
        [-4.01839525e00],
        [6.47411579e00],
        [-1.41184706e01],
        [1.90285723e01],
        [-1.24202456e01],
        [8.07076404e-01],
        [1.02797577e01],
        [-6.31421406e00],
        [-1.76244592e01],
        [4.94633937e00],
        [-3.34552627e00],
        [1.73728161e01],
        [-1.27592544e01],
        [2.42799369e-01],
        [-8.64880074e00],
        [-1.27602192e01],
        [1.34070385e01],
        [7.50089137e00],
        [-1.66766555e01],
        [-1.00130487e01],
        [-6.53155696e00],
        [1.56470458e01],
        [2.56937754e00],
        [1.80272085e00],
        [5.04460047e00],
        [5.69658275e00],
        [2.86841117e00],
        [9.32290311e00],
        [-1.61419835e01],
        [-1.16058218e01],
        [-1.81766202e01],
        [6.30179408e00],
        [1.97833851e01],
        [1.97963941e01],
        [-1.11790351e01],
        [1.20053129e01],
        [1.42977056e01],
        [-1.53979363e01],
        [1.85799577e01],
        [-1.05064356e01],
        [1.99554215e01],
        [1.67930940e01],
        [-1.17270013e01],
        [2.06252813e01],
        [4.91599915e00],
        [-1.16638196e01],
        [1.43358939e01],
        [1.78749694e01],
        [-6.83031028e00],
        [-5.81544923e00],
        [-1.54602999e01],
        [1.99025727e00],
        [-1.40931154e01],
        [-1.11606855e01],
        [-1.72219925e00],
        [9.36932106e00],
        [-1.71909084e01],
        [-7.35083439e00],
        [-3.93900250e-01],
        [-5.98678677e00],
    ]

    np.testing.assert_almost_equal(ocp.v.init.init, expected)
    print(ocp.v.init.init)


@pytest.mark.parametrize(
    "magnitude, raised_str",
    [
        (None, "'magnitude' must be specified to generate noised initial guess"),
        (tuple(np.ones(1) * 0.1), "'magnitude' must be an instance of int, float, list, or ndarray"),
        ([0.1, 0.1], "Invalid size of 'magnitude', 'magnitude' as list must be size 1 or 3"),
        (np.ones((2, 2)) * 0.1, "'magnitude' must be a 1 dimension array'"),
        (np.ones(2) * 0.1, f"Invalid size of 'magnitude', 'magnitude' as array must be size 1 or 3"),
    ],
)
def test_add_wrong_magnitude(magnitude, raised_str):

    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
    )
    biorbd_model = biorbd.Model(bioptim_folder + "/models/cube.bioMod")
    n_shooting = [20, 30, 20]

    nb_phases = ocp.n_phases

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    x_init = InitialGuessList()
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])

    with pytest.raises(ValueError, match=raised_str):
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=magnitude,  # n phase
            n_shooting=[ns for ns in n_shooting],
            bound_push=np.array((0.1)),
            seed=42,
            magnitude_type=MagnitudeType.RELATIVE,
        )


@pytest.mark.parametrize(
    "bound_push, raised_str",
    [
        (None, "'bound_push' must be specified to generate noised initial guess"),
        (tuple(np.ones(1) * 0.1), "'bound_push' must be an instance of int, float, list or ndarray"),
        ([0.1, 0.1], f"Invalid size of 'bound_push', 'bound_push' as list must be size 1 or 3"),
        (np.ones((2, 2)) * 0.1, "'bound_push' must be a 1 dimension array'"),
        (np.ones(2) * 0.1, f"Invalid size of 'bound_push', 'bound_push' as array must be size 1 or 3"),
    ],
)
def test_add_wrong_bound_push(bound_push, raised_str):

    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
    )
    biorbd_model = biorbd.Model(bioptim_folder + "/models/cube.bioMod")
    n_shooting = [20, 30, 20]

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    x_init = InitialGuessList()
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])

    with pytest.raises(ValueError, match=raised_str):
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.1,  # n phase
            n_shooting=[ns for ns in n_shooting],
            bound_push=bound_push,
            seed=42,
            magnitude_type=MagnitudeType.RELATIVE,
        )


@pytest.mark.parametrize(
    "seed, raised_str",
    [
        (0.1, "Seed must be an integer or a list of integer"),
        ([0.1, 0.1], f"Seed as list must have length = 1 or 3"),
    ],
)
def test_add_wrong_seed(seed, raised_str):

    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
    )
    biorbd_model = biorbd.Model(bioptim_folder + "/models/cube.bioMod")
    n_shooting = [20, 30, 20]

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    x_init = InitialGuessList()
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])

    with pytest.raises(ValueError, match=raised_str):
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.1,  # n phase
            n_shooting=[ns for ns in n_shooting],
            bound_push=0.1,
            seed=seed,
            magnitude_type=MagnitudeType.RELATIVE,
        )


def test_add_wrong_bounds():

    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
    )
    biorbd_model = biorbd.Model(bioptim_folder + "/models/cube.bioMod")
    n_shooting = [20, 30, 20]

    nb_phases = ocp.n_phases

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    x_init = InitialGuessList()
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])

    with pytest.raises(ValueError, match="bounds must be specified to generate noised initial guess"):
        x_init.add_noise(
            bounds=None,
            magnitude=0.1,  # n phase
            n_shooting=[ns for ns in n_shooting],
            bound_push=0.1,
            seed=42,
            magnitude_type=MagnitudeType.RELATIVE,
        )
    with pytest.raises(ValueError, match=f"Invalid size of 'bounds', 'bounds' must be size {nb_phases}"):
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.1,  # n phase
            n_shooting=[ns for ns in n_shooting],
            bound_push=0.1,
            seed=42,
            magnitude_type=MagnitudeType.RELATIVE,
        )


def test_add_wrong_n_shooting():

    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
    )
    biorbd_model = biorbd.Model(bioptim_folder + "/models/cube.bioMod")

    nb_phases = ocp.n_phases

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    x_init = InitialGuessList()
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])
    x_init.add([1, 2, 1, 2, 1, 2])

    with pytest.raises(ValueError, match="n_shooting must be specified to generate noised initial guess"):
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.1,  # n phase
            bound_push=0.1,
            seed=42,
            magnitude_type=MagnitudeType.RELATIVE,
        )
    with pytest.raises(ValueError, match=f"Invalid size of 'n_shooting', 'n_shooting' must be size {nb_phases}"):
        x_init.add_noise(
            bounds=x_bounds,
            n_shooting=[20, 30],
            magnitude=0.1,  # n phase
            bound_push=0.1,
            seed=42,
            magnitude_type=MagnitudeType.RELATIVE,
        )


def test_double_update_bounds_and_init():
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0)

    x_bounds = Bounds(-np.ones((nq * 2, 1)), np.ones((nq * 2, 1)))
    u_bounds = Bounds(-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1)))
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.array([[-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.min, expected)
    expected = np.array([[1] * (nq * 2) * (ns + 1) + [2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.max, expected)

    x_init = InitialGuess(0.5 * np.ones((nq * 2, 1)))
    u_init = InitialGuess(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.init.init, expected)

    x_bounds = Bounds(-2.0 * np.ones((nq * 2, 1)), 2.0 * np.ones((nq * 2, 1)))
    u_bounds = Bounds(-4.0 * np.ones((nq, 1)), 4.0 * np.ones((nq, 1)))
    ocp.update_bounds(x_bounds=x_bounds)
    ocp.update_bounds(u_bounds=u_bounds)

    expected = np.array([[-2] * (nq * 2) * (ns + 1) + [-4] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.min, expected)
    expected = np.array([[2] * (nq * 2) * (ns + 1) + [4] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.max, expected)

    x_init = InitialGuess(0.25 * np.ones((nq * 2, 1)))
    u_init = InitialGuess(-0.25 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.25] * (nq * 2) * (ns + 1) + [-0.25] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.init.init, expected)

    with pytest.raises(RuntimeError, match="x_init should be built from a InitialGuess or InitialGuessList"):
        ocp.update_initial_guess(x_bounds, u_bounds)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


def test_update_bounds_and_init_with_param():
    def my_parameter_function(biorbd_model, value, extra_value):
        new_gravity = MX.zeros(3, 1)
        new_gravity[2] = value + extra_value
        biorbd_model.setGravity(new_gravity)

    def my_target_function(ocp, value, target_value):
        return value + target_value

    biorbd_model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    ns = 10
    g_min, g_max, g_init = -10, -6, -8

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    parameters = ParameterList()
    bounds_gravity = Bounds(g_min, g_max, interpolation=InterpolationType.CONSTANT)
    initial_gravity = InitialGuess(g_init)
    parameter_objective_functions = Objective(
        my_target_function, weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target_value=-8
    )
    parameters.add(
        "gravity_z",
        my_parameter_function,
        initial_gravity,
        bounds_gravity,
        size=1,
        penalty_list=parameter_objective_functions,
        extra_value=1,
    )

    ocp = OptimalControlProgram(biorbd_model, dynamics, ns, 1.0, parameters=parameters)

    x_bounds = Bounds(-np.ones((nq * 2, 1)), np.ones((nq * 2, 1)))
    u_bounds = Bounds(-2.0 * np.ones((nq, 1)), 2.0 * np.ones((nq, 1)))
    ocp.update_bounds(x_bounds, u_bounds)

    expected = np.array([[-1] * (nq * 2) * (ns + 1) + [-2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.min, np.append(expected, [g_min])[:, np.newaxis])
    expected = np.array([[1] * (nq * 2) * (ns + 1) + [2] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.bounds.max, np.append(expected, [g_max])[:, np.newaxis])

    x_init = InitialGuess(0.5 * np.ones((nq * 2, 1)))
    u_init = InitialGuess(-0.5 * np.ones((nq, 1)))
    ocp.update_initial_guess(x_init, u_init)
    expected = np.array([[0.5] * (nq * 2) * (ns + 1) + [-0.5] * nq * ns]).T
    np.testing.assert_almost_equal(ocp.v.init.init, np.append(expected, [g_init])[:, np.newaxis])


def test_add_wrong_param():
    g_min, g_max, g_init = -10, -6, -8

    def my_parameter_function(biorbd_model, value, extra_value):
        biorbd_model.setGravity(biorbd.Vector3d(0, 0, value + extra_value))

    def my_target_function(ocp, value, target_value):
        return value + target_value

    parameters = ParameterList()
    initial_gravity = InitialGuess(g_init)
    bounds_gravity = Bounds(g_min, g_max, interpolation=InterpolationType.CONSTANT)
    parameter_objective_functions = Objective(
        my_target_function, weight=10, quadratic=True, custom_type=ObjectiveFcn.Parameter, target_value=-8
    )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            [],
            initial_gravity,
            bounds_gravity,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            my_parameter_function,
            None,
            bounds_gravity,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            my_parameter_function,
            initial_gravity,
            None,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    with pytest.raises(
        RuntimeError, match="function, initial_guess, bounds and size are mandatory elements to declare a parameter"
    ):
        parameters.add(
            "gravity_z",
            my_parameter_function,
            initial_gravity,
            bounds_gravity,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.SPLINE,
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        InterpolationType.EACH_FRAME,
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_init_rk4(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(
        biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=OdeSolver.RK4()
    )

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.EACH_FRAME:
        x = np.zeros((nq * 2, ns + 1))
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.random.random((nq + nqdot, ns + 1))
        u = np.random.random((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.random.random((nq + nqdot, ns + 1))
        u = np.random.random((ntau, ns))
    elif interpolation == InterpolationType.SPLINE:
        # Bound , assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))

    np.random.seed(0)
    x_init = NoisedInitialGuess(
        initial_guess=x,
        t=t,
        interpolation=interpolation,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        u,
        t=t,
        interpolation=interpolation,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns - 1,
        bound_push=0.1,
        **extra_params_u,
    )

    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)
        print(ocp.v.init.init)

        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )

        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    [1.00292881e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [1.26291136e00],
                    [5.83576452e-03],
                    [3.77852829e-01],
                    [5.34820509e-01],
                    [4.17982425e-01],
                    [3.75933739e-01],
                    [1.50616580e00],
                    [-2.49651155e-03],
                    [8.21659249e-01],
                    [-5.39051987e-01],
                    [3.49542082e-01],
                    [-4.84064610e-02],
                    [1.75269299e00],
                    [0.00000000e00],
                    [1.67000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [-7.69022965e-02],
                    [1.15886757e01],
                    [1.33822245e00],
                    [1.04301742e00],
                    [9.89739329e00],
                    [1.85693476e00],
                    [-1.90992018e00],
                    [9.46864776e00],
                    [-9.35398671e-01],
                ]
            )

        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    [0.61502453],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [0.76732112],
                    [0.43533947],
                    [0.19273203],
                    [1.15035619],
                    [0.76088147],
                    [0.81430269],
                    [0.90922359],
                    [0.13708974],
                    [0.32886699],
                    [-0.32665397],
                    [0.78933071],
                    [0.15428881],
                    [0.57293724],
                    [-0.1],
                    [1.67],
                    [-0.1],
                    [-0.1],
                    [-0.1],
                    [-0.70590907],
                    [2.24732687],
                    [-0.65897059],
                    [1.08074616],
                    [0.85131915],
                    [1.31781816],
                    [-1.21759843],
                    [0.30813958],
                    [-0.03112013],
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    [1.00292881e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [1.51291136e00],
                    [5.83576452e-03],
                    [7.70352829e-01],
                    [5.34820509e-01],
                    [4.17982425e-01],
                    [3.75933739e-01],
                    [1.50616580e00],
                    [-2.49651155e-03],
                    [8.21659249e-01],
                    [-5.39051987e-01],
                    [3.49542082e-01],
                    [-4.84064610e-02],
                    [1.50269299e00],
                    [0.00000000e00],
                    [1.67000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [0.00000000e00],
                    [-7.69022965e-02],
                    [1.15886757e01],
                    [1.33822245e00],
                    [5.59684085e-01],
                    [9.89739329e00],
                    [1.09693476e00],
                    [-1.42658685e00],
                    [9.46864776e00],
                    [-1.75398671e-01],
                ]
            )

        if interpolation == InterpolationType.EACH_FRAME:
            expected = np.array(
                [
                    [0.00292881],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.01291136],
                    [0.00583576],
                    [-0.01464717],
                    [0.53482051],
                    [0.41798243],
                    [0.37593374],
                    [0.0061658],
                    [-0.00249651],
                    [0.03665925],
                    [-0.53905199],
                    [0.34954208],
                    [-0.04840646],
                    [0.00269299],
                    [0.0],
                    [1.67],
                    [0.0],
                    [0.0],
                    [0.0],
                    [-1.5269023],
                    [1.77867567],
                    [-0.94177755],
                    [0.55968409],
                    [0.08739329],
                    [1.09693476],
                    [-1.42658685],
                    [-0.34135224],
                    [-0.17539867],
                ]
            )

        np.testing.assert_almost_equal(ocp.v.init.init, expected)

        with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
            ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.SPLINE,
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        InterpolationType.EACH_FRAME,
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_init_collocation(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif interpolation == InterpolationType.EACH_FRAME:
        x = np.zeros((nq + nqdot, ns + 1))
        for i in range(nq + nqdot):
            x[i, :] = np.linspace(i, i + 1, ns + 1)
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq + nqdot, ns * (solver.polynomial_degree + 1) + 1))
        for i in range(nq + nqdot):
            x[i, :] = np.linspace(i, i + 1, ns * (solver.polynomial_degree + 1) + 1)
        u = np.zeros((ntau, ns))
    elif interpolation == InterpolationType.SPLINE:
        # Bound , assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))

    np.random.seed(0)
    x_init = NoisedInitialGuess(
        initial_guess=x,
        t=t,
        interpolation=interpolation,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        u,
        t=t,
        interpolation=interpolation,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns - 1,
        bound_push=0.1,
        **extra_params_u,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)

    if interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                [0.00292881],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.01291136],
                [0.00583576],
                [-0.01464717],
                [0.53482051],
                [0.41798243],
                [0.37593374],
                [0.01291136],
                [0.00583576],
                [-0.01464717],
                [0.53482051],
                [0.41798243],
                [0.37593374],
                [0.01291136],
                [0.00583576],
                [-0.01464717],
                [0.53482051],
                [0.41798243],
                [0.37593374],
                [0.0061658],
                [-0.00249651],
                [0.03665925],
                [-0.53905199],
                [0.34954208],
                [-0.04840646],
                [0.0061658],
                [-0.00249651],
                [0.03665925],
                [-0.53905199],
                [0.34954208],
                [-0.04840646],
                [0.00269299],
                [0.0],
                [1.67],
                [0.0],
                [0.0],
                [0.0],
                [-1.5269023],
                [1.77867567],
                [-0.94177755],
                [0.55968409],
                [0.08739329],
                [1.09693476],
                [-1.42658685],
                [-0.34135224],
                [-0.17539867],
            ]
        )

    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                [1.00292881e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.26291136e00],
                [5.83576452e-03],
                [3.77852829e-01],
                [5.34820509e-01],
                [4.17982425e-01],
                [3.75933739e-01],
                [1.26291136e00],
                [5.83576452e-03],
                [3.77852829e-01],
                [5.34820509e-01],
                [4.17982425e-01],
                [3.75933739e-01],
                [1.26291136e00],
                [5.83576452e-03],
                [3.77852829e-01],
                [5.34820509e-01],
                [4.17982425e-01],
                [3.75933739e-01],
                [1.50616580e00],
                [-2.49651155e-03],
                [8.21659249e-01],
                [-5.39051987e-01],
                [3.49542082e-01],
                [-4.84064610e-02],
                [1.50616580e00],
                [-2.49651155e-03],
                [8.21659249e-01],
                [-5.39051987e-01],
                [3.49542082e-01],
                [-4.84064610e-02],
                [1.75269299e00],
                [0.00000000e00],
                [1.67000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [-7.69022965e-02],
                [1.15886757e01],
                [1.33822245e00],
                [1.04301742e00],
                [9.89739329e00],
                [1.85693476e00],
                [-1.90992018e00],
                [9.46864776e00],
                [-9.35398671e-01],
            ]
        )

    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                [0.61502453],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.1],
                [0.76732112],
                [0.43533947],
                [0.19273203],
                [1.15035619],
                [0.76088147],
                [0.81430269],
                [0.76732112],
                [0.43533947],
                [0.19273203],
                [1.15035619],
                [0.76088147],
                [0.81430269],
                [0.76732112],
                [0.43533947],
                [0.19273203],
                [1.15035619],
                [0.76088147],
                [0.81430269],
                [0.90922359],
                [0.13708974],
                [0.32886699],
                [-0.32665397],
                [0.78933071],
                [0.15428881],
                [0.90922359],
                [0.13708974],
                [0.32886699],
                [-0.32665397],
                [0.78933071],
                [0.15428881],
                [0.57293724],
                [-0.1],
                [1.67],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.70590907],
                [2.24732687],
                [-0.65897059],
                [1.08074616],
                [0.85131915],
                [1.31781816],
                [-1.21759843],
                [0.30813958],
                [-0.03112013],
            ]
        )

    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                [1.00292881e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [1.51291136e00],
                [5.83576452e-03],
                [7.70352829e-01],
                [5.34820509e-01],
                [4.17982425e-01],
                [3.75933739e-01],
                [1.51291136e00],
                [5.83576452e-03],
                [7.70352829e-01],
                [5.34820509e-01],
                [4.17982425e-01],
                [3.75933739e-01],
                [1.51291136e00],
                [5.83576452e-03],
                [7.70352829e-01],
                [5.34820509e-01],
                [4.17982425e-01],
                [3.75933739e-01],
                [1.50616580e00],
                [-2.49651155e-03],
                [8.21659249e-01],
                [-5.39051987e-01],
                [3.49542082e-01],
                [-4.84064610e-02],
                [1.50616580e00],
                [-2.49651155e-03],
                [8.21659249e-01],
                [-5.39051987e-01],
                [3.49542082e-01],
                [-4.84064610e-02],
                [1.50269299e00],
                [0.00000000e00],
                [1.67000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [0.00000000e00],
                [-7.69022965e-02],
                [1.15886757e01],
                [1.33822245e00],
                [5.59684085e-01],
                [9.89739329e00],
                [1.09693476e00],
                [-1.42658685e00],
                [9.46864776e00],
                [-1.75398671e-01],
            ]
        )

    elif interpolation == InterpolationType.EACH_FRAME:
        expected = np.array(
            [
                [2.92881024e-03],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [3.46244695e-01],
                [9.00000000e-01],
                [2.31868616e00],
                [3.86815384e00],
                [4.75131576e00],
                [5.70926707e00],
                [3.46244695e-01],
                [9.00000000e-01],
                [2.31868616e00],
                [3.86815384e00],
                [4.75131576e00],
                [5.70926707e00],
                [3.46244695e-01],
                [9.00000000e-01],
                [2.31868616e00],
                [3.86815384e00],
                [4.75131576e00],
                [5.70926707e00],
                [6.72832469e-01],
                [9.00000000e-01],
                [2.70332592e00],
                [3.12761468e00],
                [5.01620875e00],
                [5.61826021e00],
                [6.72832469e-01],
                [9.00000000e-01],
                [2.70332592e00],
                [3.12761468e00],
                [5.01620875e00],
                [5.61826021e00],
                [1.00269299e00],
                [-1.00000000e-01],
                [1.47000000e00],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.52690230e00],
                [1.77867567e00],
                [-9.41777552e-01],
                [5.59684085e-01],
                [8.73932870e-02],
                [1.09693476e00],
                [-1.42658685e00],
                [-3.41352240e-01],
                [-1.75398671e-01],
            ]
        )

    elif interpolation == InterpolationType.ALL_POINTS:
        expected = np.array(
            [
                [2.92881024e-03],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [1.79578029e-01],
                [9.00000000e-01],
                [2.11478380e00],
                [3.11826021e00],
                [4.05942770e00],
                [5.30753031e00],
                [3.39499136e-01],
                [9.00000000e-01],
                [2.27304220e00],
                [3.68585669e00],
                [4.03746519e00],
                [5.48027693e00],
                [5.02692991e-01],
                [9.00000000e-01],
                [2.54179824e00],
                [3.02030950e00],
                [4.84461222e00],
                [6.05763028e00],
                [6.62085955e-01],
                [9.00000000e-01],
                [2.70162087e00],
                [3.84249661e00],
                [4.61156355e00],
                [5.89514879e00],
                [8.42086980e-01],
                [9.00000000e-01],
                [2.87983043e00],
                [3.38515786e00],
                [4.91932997e00],
                [5.65678575e00],
                [9.96255233e-01],
                [-1.00000000e-01],
                [1.47000000e00],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [-1.00000000e-01],
                [7.90524784e-01],
                [6.82551478e-01],
                [-7.38286596e-01],
                [-1.75909811e00],
                [-1.15846976e00],
                [-5.45156916e-01],
                [6.67066862e-01],
                [-1.48429481e00],
                [2.80787082e-01],
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.SPLINE,
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        InterpolationType.EACH_FRAME,
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_initial_guess_rk4(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = InitialGuess([0] * (nq + nqdot), interpolation=interpolation)
        u = InitialGuess([tau_init] * ntau, interpolation=interpolation)
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = InitialGuess(
            np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T,
            interpolation=interpolation,
        )
        u = InitialGuess(
            np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation
        )
    elif interpolation == InterpolationType.LINEAR:
        x = InitialGuess(np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T, interpolation=interpolation)
        u = InitialGuess(np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.EACH_FRAME:
        x = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x[i, :] = np.linspace(i, i + 1, ns + 1)
        x = InitialGuess(x, interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq * 2, ns + 1))
        for i in range(ns + 1):
            x[i, :] = np.linspace(i, i + 1, ns + 1)
        x = InitialGuess(x, interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        np.random.seed(42)
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = InitialGuess(np.random.random((nq + nqdot, 5)), interpolation=interpolation, t=t)
        u = InitialGuess(np.random.random((ntau, 5)), interpolation=interpolation, t=t)

    x_init = NoisedInitialGuess(
        initial_guess=x,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns - 1,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )
    if interpolation == InterpolationType.ALL_POINTS:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            ocp.update_initial_guess(x_init, u_init)
    else:
        ocp.update_initial_guess(x_init, u_init)
        print(ocp.v.init.init)
        if interpolation == InterpolationType.CONSTANT:
            expected = np.array(
                [
                    0.7962362,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    0.81352143,
                    -0.00688011,
                    0.01307359,
                    -0.18074267,
                    0.01555492,
                    -0.22651269,
                    0.80695982,
                    -0.00883833,
                    -0.03012256,
                    -0.19991527,
                    -0.04276021,
                    -0.13059937,
                    0.80295975,
                    -0.1,
                    1.47,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.25091976,
                    0.19731697,
                    -0.88383278,
                    0.90142861,
                    -0.68796272,
                    0.73235229,
                    0.46398788,
                    -0.68801096,
                    0.20223002,
                ]
            )
        elif interpolation == InterpolationType.LINEAR:
            expected = np.array(
                [
                    1.79623620e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    2.06352143e00,
                    -6.88010959e-03,
                    4.05573586e-01,
                    -1.80742667e-01,
                    1.55549247e-02,
                    -2.26512688e-01,
                    2.20000000e00,
                    -8.83832776e-03,
                    7.54877435e-01,
                    -1.99915269e-01,
                    -4.27602059e-02,
                    -1.30599369e-01,
                    2.20000000e00,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    1.19908024e00,
                    1.00073170e01,
                    1.39616722e00,
                    1.38476195e00,
                    9.12203728e00,
                    1.49235229e00,
                    -1.93454497e-02,
                    9.12198904e00,
                    -5.57769977e-01,
                ]
            )
        elif interpolation == InterpolationType.SPLINE:
            expected = np.array(
                [
                    1.39489469e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    1.11672435e00,
                    6.65627498e-01,
                    2.05044956e-01,
                    1.57276580e-01,
                    4.41795628e-01,
                    1.47886691e-03,
                    9.62969993e-01,
                    4.57938262e-01,
                    1.52256794e-01,
                    2.03847059e-01,
                    5.28820074e-01,
                    1.12785129e-01,
                    9.50893802e-01,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    6.97965775e-01,
                    8.81549995e-01,
                    2.54876264e-02,
                    1.86521820e00,
                    -2.20956562e-01,
                    1.06270452e00,
                    1.30112101e00,
                    -5.07835040e-01,
                    7.90965478e-01,
                ]
            )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            expected = np.array(
                [
                    1.79623620e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    2.20000000e00,
                    -6.88010959e-03,
                    7.98073586e-01,
                    -1.80742667e-01,
                    1.55549247e-02,
                    -2.26512688e-01,
                    2.20000000e00,
                    -8.83832776e-03,
                    7.54877435e-01,
                    -1.99915269e-01,
                    -4.27602059e-02,
                    -1.30599369e-01,
                    2.20000000e00,
                    -1.00000000e-01,
                    1.47000000e00,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    -1.00000000e-01,
                    1.19908024e00,
                    1.00073170e01,
                    1.39616722e00,
                    9.01428613e-01,
                    9.12203728e00,
                    7.32352292e-01,
                    4.63987884e-01,
                    9.12198904e00,
                    2.02230023e-01,
                ]
            )

        if interpolation == InterpolationType.EACH_FRAME:
            expected = np.array(
                [
                    0.7962362,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.1,
                    1.14685476,
                    0.9,
                    2.34640692,
                    3.15259067,
                    0.01555492,
                    -0.22651269,
                    1.47362648,
                    0.9,
                    2.6365441,
                    3.4667514,
                    -0.04276021,
                    -0.13059937,
                    1.80295975,
                    -0.1,
                    1.47,
                    -0.1,
                    -0.1,
                    -0.1,
                    -0.25091976,
                    0.19731697,
                    -0.88383278,
                    0.90142861,
                    -0.68796272,
                    0.73235229,
                    0.46398788,
                    -0.68801096,
                    0.20223002,
                ]
            )

        np.testing.assert_almost_equal(ocp.v.init.init, expected[:, np.newaxis])

        with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
            ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize("n_extra", [0, 1])
def test_update_noised_initial_guess_rk4(n_extra):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0.3
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    x = InitialGuess([1] * (nq + nqdot), interpolation=InterpolationType.CONSTANT)
    u = InitialGuess([tau_init] * ntau, interpolation=InterpolationType.CONSTANT)

    state_noise = np.array([0.01] * nq + [0.2] * nqdot + [0.1] * n_extra)
    if n_extra > 0:
        with pytest.raises(
            ValueError, match="noise_magnitude must be a float or list of float of the size of states or controls"
        ):
            NoisedInitialGuess(
                initial_guess=x,
                bounds=x_bounds,
                magnitude=state_noise,
                magnitude_type=MagnitudeType.RELATIVE,
                n_shooting=ns,
                bound_push=0.1,
                seed=42,
                **extra_params_x,
            )
        return
    else:
        x_init = NoisedInitialGuess(
            initial_guess=x,
            bounds=x_bounds,
            magnitude=state_noise,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=ns,
            bound_push=0.1,
            seed=42,
            **extra_params_x,
        )

    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        magnitude=np.array([0.03] * ntau),
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns - 1,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)
    expected = np.array(
        [
            [0.99247241],
            [-0.1],
            [-0.1],
            [-0.1],
            [-0.1],
            [-0.1],
            [1.02704286],
            [0.98623978],
            [1.02614717],
            [-6.22970669],
            [1.62219699],
            [-8.06050751],
            [1.01391964],
            [0.98232334],
            [0.93975487],
            [-6.99661076],
            [-0.71040824],
            [-4.22397476],
            [1.00591951],
            [-0.1],
            [1.67],
            [-0.1],
            [-0.1],
            [-0.1],
            [-1.20551857],
            [1.48390181],
            [-5.00299665],
            [5.70857168],
            [-3.82777631],
            [4.69411375],
            [3.0839273],
            [-3.82806576],
            [1.51338014],
        ]
    )

    np.testing.assert_almost_equal(ocp.v.init.init, expected)

    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.SPLINE,
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        InterpolationType.EACH_FRAME,
        InterpolationType.ALL_POINTS,
    ],
)
def test_update_noised_initial_guess_collocation(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if interpolation == InterpolationType.CONSTANT:
        x = InitialGuess([0] * (nq + nqdot), interpolation=interpolation)
        u = InitialGuess([tau_init] * ntau, interpolation=interpolation)
    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = InitialGuess(
            np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T,
            interpolation=interpolation,
        )
        u = InitialGuess(
            np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation
        )
    elif interpolation == InterpolationType.LINEAR:
        x = InitialGuess(np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T, interpolation=interpolation)
        u = InitialGuess(np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T, interpolation=interpolation)
    elif interpolation == InterpolationType.EACH_FRAME:
        x = np.zeros((nq * 2, ns + 1))
        for i in range(nq * 2):
            x[i, :] = np.linspace(0, 1, ns + 1)
        x = InitialGuess(x, interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.ALL_POINTS:
        x = np.zeros((nq * 2, ns * (solver.polynomial_degree + 1) + 1))
        for i in range(nq * 2):
            x[i, :] = np.linspace(0, 1, ns * (solver.polynomial_degree + 1) + 1)
        x = InitialGuess(x, interpolation=interpolation)
        u = InitialGuess(np.zeros((ntau, ns)), interpolation=interpolation)
    elif interpolation == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        np.random.seed(42)
        t = np.hstack((0, np.sort(np.random.random((3,)) * phase_time), phase_time))
        x = InitialGuess(np.random.random((nq + nqdot, 5)), interpolation=interpolation, t=t)
        u = InitialGuess(np.random.random((ntau, 5)), interpolation=interpolation, t=t)

    x_init = NoisedInitialGuess(
        initial_guess=x,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
        **extra_params_x,
    )
    u_init = NoisedInitialGuess(
        initial_guess=u,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns - 1,
        bound_push=0.1,
        seed=42,
        **extra_params_u,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)
    if interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                [-0.00752759],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.02704286],
                [-0.01376022],
                [0.02614717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [0.02704286],
                [-0.01376022],
                [0.02614717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [0.02704286],
                [-0.01376022],
                [0.02614717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [0.01391964],
                [-0.01767666],
                [-0.06024513],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [0.01391964],
                [-0.01767666],
                [-0.06024513],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [0.00591951],
                [0.0],
                [1.67],
                [0.0],
                [0.0],
                [0.0],
                [-0.50183952],
                [0.39463394],
                [-1.76766555],
                [1.80285723],
                [-1.37592544],
                [1.46470458],
                [0.92797577],
                [-1.37602192],
                [0.40446005],
            ]
        )

    elif interpolation == InterpolationType.LINEAR:
        expected = np.array(
            [
                [0.99247241],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.27704286],
                [-0.01376022],
                [0.41864717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [1.27704286],
                [-0.01376022],
                [0.41864717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [1.27704286],
                [-0.01376022],
                [0.41864717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [1.51391964],
                [-0.01767666],
                [0.72475487],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [1.51391964],
                [-0.01767666],
                [0.72475487],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [1.75591951],
                [0.0],
                [1.67],
                [0.0],
                [0.0],
                [0.0],
                [0.94816048],
                [10.20463394],
                [0.51233445],
                [2.28619056],
                [8.43407456],
                [2.22470458],
                [0.44464243],
                [8.43397808],
                [-0.35553995],
            ]
        )

    elif interpolation == InterpolationType.SPLINE:
        expected = np.array(
            [
                [0.59113089],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.1],
                [0.33024578],
                [0.65874739],
                [0.21811854],
                [-0.02346609],
                [0.45735055],
                [-0.22503382],
                [0.33024578],
                [0.65874739],
                [0.21811854],
                [-0.02346609],
                [0.45735055],
                [-0.22503382],
                [0.33024578],
                [0.65874739],
                [0.21811854],
                [-0.02346609],
                [0.45735055],
                [-0.22503382],
                [0.16992981],
                [0.44909993],
                [0.12213423],
                [0.00393179],
                [0.48605987],
                [-0.01781424],
                [0.16992981],
                [0.44909993],
                [0.12213423],
                [0.00393179],
                [0.48605987],
                [-0.01781424],
                [0.15385356],
                [-0.1],
                [1.67],
                [-0.1],
                [-0.1],
                [-0.1],
                [0.44704601],
                [1.07886696],
                [-0.85834515],
                [2.76664681],
                [-0.90891928],
                [1.79505682],
                [1.76510889],
                [-1.195846],
                [0.9931955],
            ]
        )

    elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        expected = np.array(
            [
                [0.99247241],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.52704286],
                [-0.01376022],
                [0.81114717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [1.52704286],
                [-0.01376022],
                [0.81114717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [1.52704286],
                [-0.01376022],
                [0.81114717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [1.51391964],
                [-0.01767666],
                [0.72475487],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [1.51391964],
                [-0.01767666],
                [0.72475487],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [1.50591951],
                [0.0],
                [1.67],
                [0.0],
                [0.0],
                [0.0],
                [0.94816048],
                [10.20463394],
                [0.51233445],
                [1.80285723],
                [8.43407456],
                [1.46470458],
                [0.92797577],
                [8.43397808],
                [0.40446005],
            ]
        )

    elif interpolation == InterpolationType.EACH_FRAME:
        expected = np.array(
            [
                [-0.00752759],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.36037619],
                [0.31957311],
                [0.3594805],
                [-0.028152],
                [0.36444318],
                [-0.11969204],
                [0.36037619],
                [0.31957311],
                [0.3594805],
                [-0.028152],
                [0.36444318],
                [-0.11969204],
                [0.36037619],
                [0.31957311],
                [0.3594805],
                [-0.028152],
                [0.36444318],
                [-0.11969204],
                [0.6805863],
                [0.64899001],
                [0.60642154],
                [0.26683613],
                [0.58114625],
                [0.40546793],
                [0.6805863],
                [0.64899001],
                [0.60642154],
                [0.26683613],
                [0.58114625],
                [0.40546793],
                [1.00591951],
                [-0.1],
                [1.67],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.50183952],
                [0.39463394],
                [-1.76766555],
                [1.80285723],
                [-1.37592544],
                [1.46470458],
                [0.92797577],
                [-1.37602192],
                [0.40446005],
            ]
        )

    elif interpolation == InterpolationType.ALL_POINTS:
        expected = np.array(
            [
                [-0.00752759],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.19370953],
                [0.17071127],
                [0.1268821],
                [-0.09453207],
                [-0.40328055],
                [-0.07886291],
                [0.34725297],
                [0.34165624],
                [0.30873369],
                [0.16539867],
                [0.46847818],
                [-0.1722468],
                [0.50591951],
                [0.48082338],
                [0.50311098],
                [0.44479591],
                [0.0859684],
                [0.73151405],
                [0.64602779],
                [0.68546306],
                [0.65811463],
                [1.02502935],
                [0.12009438],
                [0.59146007],
                [0.812693],
                [0.84663104],
                [0.80709841],
                [0.45593228],
                [1.39741954],
                [0.35837257],
                [0.97348502],
                [-0.1],
                [1.67],
                [-0.1],
                [-0.1],
                [-0.1],
                [-0.50183952],
                [0.39463394],
                [-1.76766555],
                [1.80285723],
                [-1.37592544],
                [1.46470458],
                [0.92797577],
                [-1.37602192],
                [0.40446005],
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)


@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationType.CONSTANT,
    ],
)
def test_update_noised_initial_guess_list(interpolation):
    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model = biorbd.Model(bioptim_folder + "/examples/getting_started/models/cube.bioMod")
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    ns = 3
    phase_time = 1.0
    solver = OdeSolver.COLLOCATION(polynomial_degree=1)

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, n_shooting=ns, phase_time=phase_time, ode_solver=solver)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    x_init = NoisedInitialGuess(
        initial_guess=[0] * (nq + nqdot),
        interpolation=InterpolationType.CONSTANT,
        bounds=x_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns,
        bound_push=0.1,
        seed=42,
    )
    u_init = NoisedInitialGuess(
        initial_guess=[tau_init] * ntau,
        interpolation=InterpolationType.CONSTANT,
        bounds=u_bounds,
        magnitude=0.01,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=ns - 1,
        bound_push=0.1,
        seed=42,
    )

    ocp.update_initial_guess(x_init, u_init)
    print(ocp.v.init.init)
    if interpolation == InterpolationType.CONSTANT:
        expected = np.array(
            [
                [-0.00752759],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.02704286],
                [-0.01376022],
                [0.02614717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [0.02704286],
                [-0.01376022],
                [0.02614717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [0.02704286],
                [-0.01376022],
                [0.02614717],
                [-0.36148533],
                [0.03110985],
                [-0.45302538],
                [0.01391964],
                [-0.01767666],
                [-0.06024513],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [0.01391964],
                [-0.01767666],
                [-0.06024513],
                [-0.39983054],
                [-0.08552041],
                [-0.26119874],
                [0.00591951],
                [0.0],
                [1.67],
                [0.0],
                [0.0],
                [0.0],
                [-0.50183952],
                [0.39463394],
                [-1.76766555],
                [1.80285723],
                [-1.37592544],
                [1.46470458],
                [0.92797577],
                [-1.37602192],
                [0.40446005],
            ]
        )

    np.testing.assert_almost_equal(ocp.v.init.init, expected)
    with pytest.raises(RuntimeError, match="x_bounds should be built from a Bounds or BoundsList"):
        ocp.update_bounds(x_init, u_init)
