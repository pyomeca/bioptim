"""
This file is an example of how to run a bioptim Online plotting server. That said, this is usually not the way to run a 
bioptim server as it is easier to run it as an automatic multiprocess (default). This is achieved by setting 
`show_options={"type": ShowOnlineType.SERVER, "as_multiprocess": True}` in the solver options. 
If set to False, then the plotting server is mandatory.

Since the server runs usings sockets, it is possible to run the server on a different machine than the one running the
optimization. This is useful when the optimization is run on a cluster and the plotting server is run on a local machine.
"""

from bioptim import PlottingServer


def main():
    PlottingServer()


if __name__ == "__main__":
    main()
