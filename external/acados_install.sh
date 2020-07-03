#!/bin/bash

# Move to the external folder
cd ${0%/*}

# Download ACADOS if needed
if [ ! -f acados/CMakeLists.txt ]; then
  echo "Git submodules not initialized. Initializing"
  git submodule update --recursive --init
fi

# Check if everything required by the script is present
echo "Processing arguments"
echo ""
ARG1=${1:-$CONDA_PREFIX}
if [ -z "$ARG1" ]; then
  echo "  Argument 1 (CMAKE_INSTALL_PREFIX) is missing and you are not using conda."
  echo "  Please provide a path for installation"
  exit 1
fi

if [ -z "$1" ]; then
  echo "  Argument 1 (CMAKE_INSTALL_PREFIX) not provided, falling back on CONDA_PREFIX"
  echo "  CONDA_PREFIX=$CONDA_PREFIX"
  echo ""
fi

ARG2=${2:-X64_AUTOMATIC}
if [ -z "$2" ]; then
  echo "  Argument 2 (BLASFEO_TARGET) not provided, falling back on X64_AUTOMATIC"
  echo ""
fi

# Move to the build folder
echo "Compiling ACADOS"
echo ""
rm -rf acados/build/
mkdir acados/build
cd acados/build

# Run cmake
cmake . .. \
  -DACADOS_INSTALL_DIR="$ARG1"\
  -DACADOS_PYTHON=ON\
  -DACADOS_WITH_QPOASES=ON\
  -DBLASFEO_TARGET="$ARG2"\
  -DCMAKE_INSTALL_PREFIX="$ARG1"
  
make install -j$CPU_COUNT



# Prepare the Python interface
cd ../interfaces/acados_template

# Prepare some modification on the files so it works with biorbd
# Allow for any python
TO_REPLACE_PYTHON_REQUIRED="python_requires"
REPLACE_PYTHON_REQUIRED_BY="# python_requires"

# Removing the casadi dependency (already installed from biorbd)
TO_REPLACE_CASADI_DEP="'casadi"
REPLACE_CASADI_DEP_BY="# 'casadi"

# Modify relative path of acados_template is install doesn't have the 
# same structure as the source folder
TO_REPLACE_PATH="'..\/..\/..\/'"
REPLACE_PATH_BY="'..\/..\/..\/..\/'"

# Perform the modifications
sed -i "s/$TO_REPLACE_PYTHON_REQUIRED/$REPLACE_PYTHON_REQUIRED_BY/" setup.py
sed -i "s/$TO_REPLACE_CASADI_DEP/$REPLACE_CASADI_DEP_BY/" setup.py
sed -i "s/$TO_REPLACE_PATH/$REPLACE_PATH_BY/" acados_template/utils.py

# Install the Python interface
pip install .

# Undo the modifications to the files (so it is not pick up by Git)
sed -i "s/$REPLACE_PYTHON_REQUIRED_BY/$TO_REPLACE_PYTHON_REQUIRED/" setup.py
sed -i "s/$REPLACE_CASADI_DEP_BY/$TO_REPLACE_CASADI_DEP/" setup.py
sed -i "s/$REPLACE_PATH_BY/$TO_REPLACE_PATH/" acados_template/utils.py

