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

# Check if there are a number of CPUs for Acados multiprocessing
NB_CPU_MAX=`cat /proc/cpuinfo | grep processor | wc -l`
ARG1=${1:-$NB_CPU_MAX}
if [ -z "$1" ]; then
  echo "  Argument 1 (NB_CPU) not provided, falling back on maximum number of CPUs ($NB_CPU_MAX)."
fi
echo "  Number of threads for acados with openMP: NB_CPU=$ARG1"
echo ""

ARG2=${2:-$CONDA_PREFIX}
if [ -z "$ARG2" ]; then
  echo "  Argument 2 (CMAKE_INSTALL_PREFIX) is missing and you are not using conda."
  echo "  Please provide a path for installation"
  exit 1
fi
if [ -z "$2" ]; then
  echo "  Argument 2 (CMAKE_INSTALL_PREFIX) not provided, falling back on CONDA_PREFIX"
fi
echo "  set CMAKE_INSTALL_PREFIX=$ARG2"
echo ""

ARG3=${3:-X64_AUTOMATIC}
if [ -z "$3" ]; then
  echo "  Argument 3 (BLASFEO_TARGET) not provided, falling back on X64_AUTOMATIC"
fi
echo "  set BLASFEO_TARGET=$ARG3"
echo ""

# Preparing environment
if [ "$CONDA_PREFIX" ]; then
  conda install git cmake -cconda-forge -y
fi

# Move to the build folder
echo "Compiling ACADOS"
echo ""
rm -rf acados/build/
mkdir acados/build
cd acados/build

# We must manually change the minimum required cmake version in some of acados' dependencies
sed -i "s/cmake_minimum_required(VERSION 3.5)/cmake_minimum_required(VERSION 3.14)/" ../external/blasfeo/CMakeLists.txt
sed -i "s/cmake_minimum_required(VERSION 2.6)/cmake_minimum_required(VERSION 3.14)/" ../external/qpoases/CMakeLists.txt
sed -i "s/CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )/cmake_minimum_required(VERSION 3.14)/" ../external/qpdunes/CMakeLists.txt
sed -i "s/cmake_minimum_required (VERSION 3.2)/cmake_minimum_required (VERSION 3.14)/" ../external/osqp/CMakeLists.txt
sed -i "s/cmake_minimum_required (VERSION 3.2)/cmake_minimum_required (VERSION 3.14)/" ../external/osqp/lin_sys/direct/qdldl/qdldl_sources/CMakeLists.txt

# Run cmake
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$ARG2" \
  -DACADOS_INSTALL_DIR="$ARG2" \
  -DACADOS_PYTHON=ON \
  -DACADOS_WITH_QPOASES=ON \
  -DACADOS_WITH_OSQP=ON \
  -DACADOS_WITH_QPDUNES=ON \
  -DBLASFEO_TARGET="$ARG3" \
  -DACADOS_WITH_OPENMP=ON \
  -DACADOS_NUM_THREADS=$ARG1
make install -j$NB_CPU_MAX

# Prepare the Python interface
cd ../interfaces/acados_template

# Removing the casadi dependency (it will conflit the already installed one from biorbd)
TO_REPLACE_CASADI_DEP="'casadi"
REPLACE_CASADI_DEP_BY="# 'casadi"

# Changed acados path
TO_REPLACE_ACADOS_SOURCE="    ACADOS_PATH = os.environ.get('ACADOS_SOURCE_DIR')"
REPLACE_ACADOS_SOURCE_BY="    ACADOS_PATH = os.environ['CONDA_PREFIX']"

TO_REPLACE_ACADOS_PYTHON="ACADOS_PYTHON_INTERFACE_PATH = os.environ.get('ACADOS_PYTHON_INTERFACE_PATH')"
REPLACE_ACADOS_PYTHON_BY="import site\n    acados_path = site.getsitepackages()\n    ACADOS_PYTHON_INTERFACE_PATH = os.path.join(acados_path[0], 'acados_template')"

# Perform the modifications
sed -i "s/$TO_REPLACE_CASADI_DEP/$REPLACE_CASADI_DEP_BY/" setup.py
sed -i "s/$TO_REPLACE_ACADOS_PYTHON/$REPLACE_ACADOS_PYTHON_BY/" acados_template/utils.py
sed -i "s/$TO_REPLACE_ACADOS_SOURCE/$REPLACE_ACADOS_SOURCE_BY/" acados_template/utils.py

# Install the Python interface
pip install .
cd ../..

# Undo the modifications to the files (so it is not picked up by Git)
git reset --hard
