from multiprocessing import Process

from .online_callback_server import PlottingServer, OnlineCallbackServer


def _start_server_internal(**kwargs):
    """
    Starts the server (necessary for multiprocessing), this method should not be called directly, apart from
    run_as_multiprocess

    Parameters
    ----------
    same as PlottingServer
    """
    PlottingServer(**kwargs)


class OnlineCallbackMultiprocessServer(OnlineCallbackServer):
    def __init__(self, *args, **kwargs):
        """
        Starts the server in a new process

        Parameters
        ----------
        Same as PlottingServer
        """
        host = kwargs["host"] if "host" in kwargs else None
        port = kwargs["port"] if "port" in kwargs else None
        log_level = None
        if "log_level" in kwargs:
            log_level = kwargs["log_level"]
            del kwargs["log_level"]

        process = Process(target=_start_server_internal, kwargs={"host": host, "port": port, "log_level": log_level})
        process.start()

        super(OnlineCallbackMultiprocessServer, self).__init__(*args, **kwargs)
