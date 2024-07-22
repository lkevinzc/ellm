import os
import socket


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    ip, port = sock.getsockname()
    sock.close()
    return port


class DistributedLauncher:
    def __init__(
        self, world_size, rank, local_rank, master_addr, master_port, is_master=False
    ) -> None:
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr
        self._master_port = master_port
        if is_master:
            self._master_port = self.bind()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(0)

    def bind(self):
        with socket.socket() as sock:
            sock.bind((self._master_addr, self._master_port))
            return sock.getsockname()[1]
