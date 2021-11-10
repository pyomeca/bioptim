"""
Test for file IO
"""
from bioptim.examples.__main__ import ExampleLoader


def test_run_examples():
    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(loader.ui.exampleTree.model().index(0, 0))

