from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bioptim",
    version="0.0.1",
    author="Pariterre",
    author_email="pariterre@hotmail.com",
    description="bioptim is a Python optimization framework that links CasADi, ipopt and biorbd for human Optimal Control Programming",
    long_description=long_description,
    url="https://github.com/bioptim/bioptim",
    packages=["bioptim", "bioptim/dynamics", "bioptim/gui", "bioptim/interfaces", "bioptim/limits", "bioptim/misc"],
    license="LICENSE",
    keywords=["biorbd", "ipopt", "CasADi", "Optimal control"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    zip_safe=False,
)
