from enum import Enum
import pytest
import pickle


class VersionFiles(Enum):
    V_223 = "file_2_2_3.bo"
    V_222 = "file_2_2_3.bo"

# if this test doesn't work, it means bioptim as changed and the old files .bo
# are not compatible anymore thus they won't be loaded by the new version of bioptim

@pytest.mark.parametrize("file", VersionFiles)
def test_open_old_files(file: VersionFiles):
    with open(file.value, "rb") as file:
        data = pickle.load(file)
