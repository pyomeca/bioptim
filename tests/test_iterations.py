import os
import pickle
import numpy as np

from bioptim.gui.plot import Iterations


def test_iterations():
    V = [np.random.random((10, 1)) for _ in range(5)]

    directory = ".__tmp_bioptim"
    file_path = ".__tmp_bioptim/temp_save_iter.bobo"
    os.mkdir(directory)
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, "wb") as file:
        pickle.dump([], file)

    for v in V:
        Iterations.save(v)

    with open(file_path, "rb") as file:
        sol_iterations = pickle.load(file)
        os.remove(file_path)
        os.rmdir(directory)

    for i in range(len(V)):
        np.testing.assert_almost_equal(V[i], sol_iterations[i])

    for i in range(len(V)):
        np.testing.assert_almost_equal(V[i], sol_iterations[i])
