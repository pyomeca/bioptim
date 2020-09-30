from setuptools import setup

setup(
    name="bioptim",
    description="BiorbdOptim is a Python optimization framework that links CasADi, ipopt and biorbd for Optimal Control Problem ",
    author="Benjamin Michaud",
    author_email="pariterre@hotmail.com",
    url="https://github.com/BiorbdOptim/BiorbdOptim",
    license="Apache 2.0",
    packages=["bioptim"],
    keywords=["biorbd", "ipopt", "CasADi"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
