"""Reference to https://github.com/mosecorg/mosec."""

import multiprocessing as mp
import pickle
import subprocess
import time
from typing import Any

import launchpad as lp
from pyarrow import plasma  # type: ignore


class PlasmaShmServer:
    def __init__(self, size_mb: int = 5):
        self._size_mb = size_mb
        self._terminated = False
        self._shm_path = ""

    def get_shm_path(self):
        return self._shm_path

    def halt(self):
        self._terminated = True

    def _start_plasma_server(self, size_mb):
        with plasma.start_plasma_store(plasma_store_memory=size_mb * 1000 * 1000) as (
            shm_path,
            shm_process,
        ):
            self._shm_path = shm_path
            while not self._terminated:
                time.sleep(3)
                code = None
                if isinstance(shm_process, mp.Process):
                    code = shm_process.exitcode
                elif isinstance(shm_process, subprocess.Popen):
                    code = shm_process.poll()

                if code is not None:
                    print(f"Plasma daemon process error {code}")
                    break

    def run(self):
        self._start_plasma_server(self._size_mb)
        lp.stop()


class PlasmaShmClient:
    """Plasma shared memory client."""

    _plasma_client = None

    def __init__(self, server: PlasmaShmServer) -> None:
        self.server = server

    def _get_client(self):
        """Get the plasma client. This will create a new one if not exist."""

        if not self._plasma_client:
            path = self.server.get_shm_path()
            if not path:
                raise RuntimeError("plasma path no found")
            self._plasma_client = plasma.connect(path)
        return self._plasma_client

    def serialize_ipc(self, data: Any) -> bytes:
        """Save the data to the plasma server and return the id."""
        client = self._get_client()
        object_id = client.put(pickle.dumps(data))
        return object_id.binary()

    def deserialize_ipc(self, data: bytes) -> Any:
        """Get the data from the plasma server and delete it."""
        client = self._get_client()
        object_id = plasma.ObjectID(bytes(data))
        obj = pickle.loads(client.get(object_id))
        client.delete((object_id,))
        return obj
