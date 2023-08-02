"""
Test for file IO
"""
from sys import platform


def test_run_examples():
#    if platform == "linux":
#        return  # AppVeyor cannot work with graphic interface on Linux

    from bioptim.examples.__main__ import ExampleLoader

    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(loader.ui.exampleTree.model().index(0, 0))
