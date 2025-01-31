"""
This file is an example of how to run a bioptim Online plotting server. Apart on Macos, this is usually not the way
to run a bioptim server as it is easier to run it as an automatic multiprocess. This is achieved by setting
`Solver.IPOPT(online_optim=OnlineOptim.MULTIPROCESS_SERVER)`.
If set to OnlineOptim.SERVER, then the plotting server is mandatory.

Since the server runs usings sockets, it is possible to run the server on a different machine than the one running the
optimization. This is useful when the optimization is run on a cluster and the plotting server is run on a local machine.

On Macos, this server is necessary as it won't connect using multiprocess. One can simply run the current script on
another terminal to access the online graphs
"""

from bioptim import PlottingServer


def main():
    PlottingServer()


if __name__ == "__main__":
    main()
