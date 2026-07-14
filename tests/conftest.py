# tests/conftest.py

"""
Shared pytest fixtures for all test suites.

qdrant fixture (session-scoped):
  Ensures a running 'qdrant-local' Docker container AND an indexed
  'openapi_reference' collection before integration tests run.

  - If the container does not exist yet, it is created (docker run) and its
    volume persists across runs.
  - If it exists but is stopped, it is started (docker start).
  - Once Qdrant accepts connections, the OpenAPI reference collection is indexed
    if missing (index_openapi_reference() is idempotent — it skips when the
    collection already has vectors).

  Only containers this fixture started are stopped on teardown; a container that
  was already running is left running.
"""

import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

QDRANT_CONTAINER = "qdrant-local"
QDRANT_IMAGE     = "qdrant/qdrant"
QDRANT_HOST      = "localhost"
QDRANT_PORT      = 6333
STARTUP_TIMEOUT  = 30  # seconds to wait for Qdrant to accept connections


def _is_qdrant_up() -> bool:
    """
    True only when Qdrant's HTTP API is ready to serve requests.

    A plain TCP connect succeeds as soon as the port is bound, which happens
    before the Actix HTTP server accepts requests — hitting that window causes
    "Connection reset by peer". Probing /readyz avoids that race.
    """
    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=2):
            pass
    except OSError:
        return False
    try:
        with urllib.request.urlopen(
            f"http://{QDRANT_HOST}:{QDRANT_PORT}/readyz", timeout=2
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionResetError, OSError):
        return False


def _container_exists() -> bool:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{QDRANT_CONTAINER}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return QDRANT_CONTAINER in result.stdout.split()


def _wait_until_ready() -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while not _is_qdrant_up():
        if time.time() > deadline:
            raise RuntimeError(
                f"Qdrant container '{QDRANT_CONTAINER}' did not become ready "
                f"within {STARTUP_TIMEOUT}s."
            )
        time.sleep(1)


@pytest.fixture(scope="session")
def qdrant():
    """
    Ensures Qdrant is running and the OpenAPI reference collection is indexed
    for the duration of the test session. Yields once both are ready.
    """
    already_running = _is_qdrant_up()

    if not already_running:
        if _container_exists():
            # Container exists but is stopped — start it.
            subprocess.run(
                ["docker", "start", QDRANT_CONTAINER],
                check=True,
                capture_output=True,
            )
        else:
            # First run on this machine — create the container.
            subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", QDRANT_CONTAINER,
                    "-p", f"{QDRANT_PORT}:6333",
                    QDRANT_IMAGE,
                ],
                check=True,
                capture_output=True,
            )
        _wait_until_ready()

    # Ensure the OpenAPI reference collection exists. index_openapi_reference()
    # is idempotent: it skips when the collection is already populated. Retry a
    # few times in case Qdrant drops the first request while still warming up.
    from openapi_rulesbank.tools.rag_tools import index_openapi_reference
    for attempt in range(3):
        try:
            index_openapi_reference()
            break
        except (ConnectionResetError, OSError):
            if attempt == 2:
                raise
            time.sleep(2)

    yield  # tests run here

    if not already_running:
        subprocess.run(
            ["docker", "stop", QDRANT_CONTAINER],
            check=True,
            capture_output=True,
        )
