import os
import platform
import time
import shutil
from pathlib import Path

import pytest

try:
    import psutil
except Exception as e:
    pytest.skip("psutil is required for these tests", allow_module_level=True)

try:
    import carla  # noqa: F401
    CARLA_AVAILABLE = True
except Exception:
    CARLA_AVAILABLE = False

# Make sure we can import from project root if running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OpenCDA.CARLA.config_yaml.conf import load_cfg  # noqa: E402
from OpenCDA.CARLA.manager import CarlaManager  # noqa: E402


def _launcher_exists(cfg) -> bool:
    if platform.system().lower().startswith("win"):
        launcher = str(cfg.windows_launcher)
    else:
        launcher = str(cfg.linux_launcher)
    return Path(launcher).is_file()


def _is_listening(port: int) -> bool:
    """Check if any process is listening on TCP port."""
    for c in psutil.net_connections(kind="tcp"):
        if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
            return True
    return False


@pytest.fixture(scope="session")
def cfg():
    """
    Load default config and apply optional environment overrides.
    Examples:
      export CARLA_LAUNCHER_LINUX=/opt/carla-simulator/CarlaUE4.sh
      set CARLA_LAUNCHER_WINDOWS=D:\Carla 0914\CarlaUE4.exe
      export CARLA_EXTRA_ARGS="-opengl,-RenderOffScreen"
    """
    overrides = {}

    env_linux = os.getenv("CARLA_LAUNCHER_LINUX")
    env_windows = os.getenv("CARLA_LAUNCHER_WINDOWS")
    if env_linux:
        overrides["linux_launcher"] = env_linux
    if env_windows:
        overrides["windows_launcher"] = env_windows

    env_args = os.getenv("CARLA_EXTRA_ARGS")
    if env_args:
        # comma-separated list -> YAML list
        overrides["extra_args"] = [a for a in env_args.split(",") if a]

    env_port = os.getenv("CARLA_PORT")
    if env_port:
        overrides["port"] = int(env_port)

    env_timeout = os.getenv("CARLA_TIMEOUT")
    if env_timeout:
        overrides["timeout"] = float(env_timeout)

    cfg = load_cfg(overrides=overrides)

    # Optional: write server logs if a dir is provided
    log_dir = os.getenv("CARLA_LOG_DIR")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        cfg.stdout_path = str(Path(log_dir) / "carla_server_stdout.log")
        cfg.stderr_path = str(Path(log_dir) / "carla_server_stderr.log")

    return cfg


@pytest.fixture
def mgr(cfg):
    """Fresh manager per test."""
    return CarlaManager(cfg)


@pytest.mark.skipif(not CARLA_AVAILABLE, reason="CARLA Python API not installed")
def test_launcher_present(cfg):
    """Ensure the configured launcher path exists for the current OS."""
    assert _launcher_exists(cfg), (
        "CARLA launcher not found. Set correct path in carla/config.yaml or via "
        "CARLA_LAUNCHER_LINUX / CARLA_LAUNCHER_WINDOWS environment variables."
    )


@pytest.mark.skipif(not CARLA_AVAILABLE, reason="CARLA Python API not installed")
def test_ensure_running_connect_query_world(mgr, cfg):
    """
    ensure_running -> get client -> basic queries.
    No printing; assert invariants.
    """
    client = mgr.ensure_running()
    version = client.get_server_version()
    assert isinstance(version, str) and len(version) > 0

    world = client.get_world()
    assert world is not None

    cmap = world.get_map()
    assert cmap is not None
    assert isinstance(cmap.name, str) and len(cmap.name) > 0

    # Manager should have the port listening now
    assert _is_listening(int(cfg.port))


@pytest.mark.skipif(not CARLA_AVAILABLE, reason="CARLA Python API not installed")
def test_restart_reconnect(mgr, cfg):
    """
    Restart the server and verify we can reconnect cleanly.
    """
    client = mgr.ensure_running()
    pre_version = client.get_server_version()
    assert pre_version

    client2 = mgr.restart()
    post_version = client2.get_server_version()
    assert post_version

    # After restart, port must be listening again
    assert _is_listening(int(cfg.port))


@pytest.mark.skipif(not CARLA_AVAILABLE, reason="CARLA Python API not installed")
def test_graceful_shutdown(mgr, cfg):
    """
    Shutdown server we control, then assert port is no longer listening within a grace window.
    """
    mgr.ensure_running()
    assert _is_listening(int(cfg.port))

    mgr.shutdown(force=False)

    # Allow a short window for the process to go down
    deadline = time.time() + 10.0
    while time.time() < deadline and _is_listening(int(cfg.port)):
        time.sleep(0.5)

    assert not _is_listening(int(cfg.port)), "CARLA still listening after shutdown grace window"


@pytest.mark.skipif(not CARLA_AVAILABLE, reason="CARLA Python API not installed")
def test_force_shutdown(mgr, cfg):
    """
    Force kill (for robustness). Useful if UE hangs.
    """
    mgr.ensure_running()
    assert _is_listening(int(cfg.port))

    mgr.shutdown(force=True)

    # Allow a brief window to release the port
    deadline = time.time() + 10.0
    while time.time() < deadline and _is_listening(int(cfg.port)):
        time.sleep(0.5)

    assert not _is_listening(int(cfg.port))
