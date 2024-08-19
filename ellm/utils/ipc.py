"""Reference to https://github.com/mosecorg/mosec."""

import multiprocessing as mp
import os
import pickle
import subprocess
import time
from typing import Any

import launchpad as lp
from pyarrow import plasma  # type: ignore

_PLASMA_PATH_FILE = "./.plasma_cache"


def start_plasma_server(size_mb: int = 5):
    if os.path.exists(_PLASMA_PATH_FILE):
        os.remove(_PLASMA_PATH_FILE)
    with plasma.start_plasma_store(plasma_store_memory=size_mb * 1000 * 1000) as (
        shm_path,
        shm_process,
    ):
        open(_PLASMA_PATH_FILE, "w").write(shm_path)
        while True:
            time.sleep(3)
            code = None
            if isinstance(shm_process, mp.Process):
                code = shm_process.exitcode
            elif isinstance(shm_process, subprocess.Popen):
                code = shm_process.poll()

            if code is not None:
                print(f"Plasma daemon process error {code}")
                break


class PlasmaShmServer:
    def __init__(self, size_mb: int = 5):
        self._size_mb = size_mb

    def run(self):
        start_plasma_server(self._size_mb)
        lp.stop()


class PlasmaShmClient:
    """Plasma shared memory client."""

    _plasma_client = None

    def _get_client(self):
        """Get the plasma client. This will create a new one if not exist."""

        if not self._plasma_client:
            path = open(_PLASMA_PATH_FILE, "r").read()
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
