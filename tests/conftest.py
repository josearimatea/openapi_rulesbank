# tests/conftest.py

"""
Shared pytest fixtures for all test suites.

qdrant fixture (session-scoped):
  Starts the existing 'qdrant-local' Docker container before integration tests
  and stops it afterwards. The container and its indexed collection are assumed
  to already exist (created via scripts/index_openapi_reference.py).

  If the container is already running (e.g. started manually), it is left
  running after the session ends — only containers started by this fixture
  are stopped on teardown.
"""

import socket
import subprocess
import time

import pytest

QDRANT_CONTAINER = "qdrant-local"
QDRANT_HOST      = "localhost"
QDRANT_PORT      = 6333
STARTUP_TIMEOUT  = 30  # seconds to wait for Qdrant to accept connections


def _is_qdrant_up() -> bool:
    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def qdrant():
    """
    Ensures the Qdrant container is running for the duration of the test session.
    Yields once Qdrant is accepting connections on localhost:6333.
    """
    already_running = _is_qdrant_up()

    if not already_running:
        subprocess.run(
            ["docker", "start", QDRANT_CONTAINER],
            check=True,
            capture_output=True,
        )
        deadline = time.time() + STARTUP_TIMEOUT
        while not _is_qdrant_up():
            if time.time() > deadline:
                raise RuntimeError(
                    f"Qdrant container '{QDRANT_CONTAINER}' did not become ready "
                    f"within {STARTUP_TIMEOUT}s."
                )
            time.sleep(1)

    yield  # tests run here

    if not already_running:
        subprocess.run(
            ["docker", "stop", QDRANT_CONTAINER],
            check=True,
            capture_output=True,
        )
