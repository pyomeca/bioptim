"""
Test for file IO
"""

import os
from sys import platform


def test_example_paths_are_preserved_when_flattening(monkeypatch):
    if platform == "linux":
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from bioptim.examples.__main__ import examples_, path, unnestedDict

    example_paths = unnestedDict(examples_["toy_examples/acados"], "toy_examples/acados")

    assert example_paths["Static arm"] == "toy_examples/acados/static_arm.py"
    for root_dir, examples in examples_.items():
        for relative_path in unnestedDict(examples, root_dir).values():
            assert os.path.isfile(os.path.join(path, relative_path))


def test_run_examples():
    if platform == "linux":  # AppVeyor and GitHub action cannot work with graphic interface on Linux
        return

    from bioptim.examples.__main__ import ExampleLoader

    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(loader.ui.exampleTree.model().index(0, 0))
