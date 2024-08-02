from multiprocessing import Process

from .online_callback_server import PlottingServer, OnlineCallbackServer


def _start_as_multiprocess_internal(**kwargs):
    """
    Starts the server (necessary for multiprocessing), this method should not be called directly, apart from
    run_as_multiprocess

    Parameters
    ----------
    same as PlottingServer
    """
    PlottingServer(**kwargs)


class PlottingMultiprocessServer(OnlineCallbackServer):
    def __init__(self, *args, **kwargs):
        """
        Starts the server in a new process

        Parameters
        ----------
        Same as PlottingServer
        """
        host = kwargs["host"] if "host" in kwargs else None
        port = kwargs["port"] if "port" in kwargs else None
        process = Process(target=_start_as_multiprocess_internal, kwargs={"host": host, "port": port})
        process.start()

        super(PlottingMultiprocessServer, self).__init__(*args, **kwargs)
