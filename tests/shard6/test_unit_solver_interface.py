import pytest
from casadi import SX, MX, vertcat, Function
import numpy as np

from bioptim import NonLinearProgram, PhaseDynamics


@pytest.fixture
def nlp_sx():
    # Create a dummy NonLinearProgram object with necessary attributes
    nlp = NonLinearProgram(None)
    nlp.X = [SX([[1], [2], [3]])]
    nlp.X_scaled = [SX([[4], [5], [6]])]
    # Add more attributes as needed
    return nlp


@pytest.fixture
def nlp_mx():
    # Create a dummy NonLinearProgram object with necessary attributes
    nlp = NonLinearProgram(None)
    nlp.X = [MX(np.array([[1], [2], [3]]))]
    nlp.X_scaled = [MX(np.array([[4], [5], [6]]))]
    # Add more attributes as needed
    return nlp


@pytest.fixture
def nlp_control_sx():
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE)
    nlp.U_scaled = [SX([[1], [2], [3]])]
    return nlp


@pytest.fixture
def nlp_control_mx():
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE)
    nlp.U_scaled = [MX(np.array([[1], [2], [3]]))]
    return nlp

