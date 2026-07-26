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

    assert os.path.normpath(example_paths["Static arm"]) == os.path.normpath("toy_examples/acados/static_arm.py")
    for root_dir, examples in examples_.items():
        for relative_path in unnestedDict(examples, root_dir).values():
            assert os.path.isfile(os.path.join(path, relative_path))


def test_filter_by_content_uses_example_package_path(monkeypatch, tmp_path):
    if platform == "linux":
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    import bioptim.examples.__main__ as examples_module

    example_directory = tmp_path / "nested"
    example_directory.mkdir()
    (example_directory / "matching.py").write_text("the content needle is here")
    (example_directory / "other.py").write_text("unrelated content")

    monkeypatch.setattr(
        examples_module,
        "examples_",
        {
            "nested": {
                "Matching example": "matching.py",
                "Other example": "other.py",
            },
            "metadata": "ignored",
        },
    )
    monkeypatch.setattr(examples_module, "path", str(tmp_path))

    class Highlighter:
        searchText = None

        @staticmethod
        def setDocument(_):
            pass

    class CodeView:
        @staticmethod
        def document():
            return None

    class Loader:
        hl = Highlighter()
        ui = type("Ui", (), {"codeView": CodeView()})()
        matching_titles = None

        @staticmethod
        def getExampleContent(filename):
            with open(filename) as example_file:
                return example_file.read()

        def showExamplesByTitle(self, titles):
            self.matching_titles = titles

    loader = Loader()
    examples_module.ExampleLoader.filterByContent(loader, "NEEDLE")

    assert loader.hl.searchText == "NEEDLE"
    assert loader.matching_titles == ["Matching example"]


def test_run_examples():
    if platform == "linux":  # AppVeyor and GitHub action cannot work with graphic interface on Linux
        return

    from bioptim.examples.__main__ import ExampleLoader

    loader = ExampleLoader()
    loader.ui.exampleTree.setCurrentIndex(loader.ui.exampleTree.model().index(0, 0))
