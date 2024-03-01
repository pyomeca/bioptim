"""
Test for file IO
"""

from sys import platform


def test_run_examples():
    if platform == "linux":  # AppVeyor and GitHub action cannot work with graphic interface on Linux
        return

    from bioptim.examples.__main__ import ExampleLoader

    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(loader.ui.exampleTree.model().index(0, 0))
