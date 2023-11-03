from bioptim import __version__ as bioptim_version
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bioptim",
    version=bioptim_version,
    author="Pariterre",
    author_email="pariterre@hotmail.com",
    description="An optimization framework for Optimal Control Programming in biomechanics",
    long_description=long_description,
    url="https://github.com/bioptim/bioptim",
    packages=[
        ".",
        "bioptim",
        "bioptim/dynamics",
        "bioptim/dynamics/fatigue",
        "bioptim/gui",
        "bioptim/interfaces",
        "bioptim/limits",
        "bioptim/misc",
        "bioptim/optimization",
        "bioptim/optimization/solution",
        "bioptim/models/"
        "bioptim/models/biorbd",
        "bioptim/models/protocols",
        "examples",
    ],
    license="LICENSE",
    keywords=["biorbd", "Ipopt", "CasADi", "Optimal control", "biomechanics"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
