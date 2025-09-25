import platform
from pathlib import Path


class ExampleUtils:
    @property
    def folder(self) -> str:
        """Returns the path to the examples folder."""
        return ExampleUtils._capitalize_folder_drive(str(Path(__file__).parent))

    @staticmethod
    def _capitalize_folder_drive(folder: str) -> str:
        if platform.system() == "Windows" and folder[1] == ":":
            # Capitalize the drive letter if it is windows
            folder = folder[0].upper() + folder[1:]
        return folder
