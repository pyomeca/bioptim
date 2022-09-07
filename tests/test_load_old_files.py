from enum import Enum
import pytest
import pickle


class VersionFiles(Enum):
    V_223 = "file_2_2_3.botest"
    V_222 = "file_2_2_2.botest"


# if this test doesn't work, it means bioptim as changed and the old files .bo
# are not compatible anymore thus they won't be loaded by the new version of bioptim


@pytest.mark.parametrize("file", VersionFiles)
def test_open_old_files(file: VersionFiles):
    if file == VersionFiles.V_222:
        # mathc an error with pytest
        with pytest.raises(ValueError, match="'user' is not a valid ConstraintType"):
            with open(file.value, "rb") as file:
                data = pickle.load(file)
    else:
        with open(file.value, "rb") as file:
            data = pickle.load(file)
