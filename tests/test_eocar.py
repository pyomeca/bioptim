"""
Test for file IO
"""
from pathlib import Path

import numpy as np
import pytest

from ..examples import eocar

# Path
PROJECT_FOLDER = Path(__file__).parent / ".."


def test_oecar():
    nlp = eocar.prepare_nlp()
    sol = nlp.solve()

    for idx in range(nlp.model.nbQ()):
        q = sol["x"][0 * nlp.model.nbQ() + idx::3 * nlp.model.nbQ()]
        q_dot = sol["x"][1 * nlp.model.nbQ() + idx::3 * nlp.model.nbQ()]
        u = sol["x"][2 * nlp.model.nbQ() + idx::3 * nlp.model.nbQ()]

