# CARLA/manager.py
from __future__ import annotations

import os
import time
import signal
import platform
import subprocess
from typing import List, Optional

try:
    import psutil  # type: ignore
except ImportError as e:
    raise SystemExit("psutil is required. Install with: pip install psutil") from e

try:
    import carla  # CARLA Python API must be installed
except ImportError:
    carla = None  # allow import for tooling; runtime connect() will require it

from omegaconf import DictConfig
from .config_yaml.conf import load_cfg  # loads CARLA/config_yaml/default.yaml (+ optional overrides)


class CarlaManager:
    """
    CARLA server lifecycle manager using YAML (OmegaConf) configuration.

    Expected YAML keys (with typical defaults):
      host: "127.0.0.1"
      port: 2000
      timeout: 20.0
      linux_launcher: "/opt/carla-simulator/CarlaUE4.sh"
      windows_launcher: "C:\\CARLA\\CarlaUE4.exe"
      extra_args: ["-opengl"]
      boot_grace_seconds: 5.0
      connect_retry_seconds: 45.0
      kill_stale_on_launch: true
      workdir: null
      stdout_path: null
      stderr_path: null
    """

    def __init__(self, cfg: Optional[DictConfig] = None):
        self.cfg: DictConfig = cfg or load_cfg()
        self.proc: Optional[subprocess.Popen] = None
        self._os = platform.system().lower()  # 'windows', 'linux', 'darwin'

    # ---------- Public API ----------

    def ensure_running(self) -> "carla.Client":
        """Ensure CARLA is listening and responsive; return a connected client."""
        if bool(self.cfg.kill_stale_on_launch):
            self._kill_stale_processes()

        if not self._is_listening(int(self.cfg.port)):
            self.launch()

        return self._wait_until_ready()

    def launch(self) -> None:
        """Launch a CARLA server process for this platform (if not already listening)."""
        if self._is_listening(int(self.cfg.port)):
            return

        launcher, args = self._launcher_cmd()
        stdout = open(self.cfg.stdout_path, "ab") if self.cfg.stdout_path else None
        stderr = open(self.cfg.stderr_path, "ab") if self.cfg.stderr_path else None

        self.proc = subprocess.Popen(
            [launcher] + args,
            cwd=self.cfg.workdir or os.path.dirname(launcher) or None,
            stdout=stdout or subprocess.PIPE,
            stderr=stderr or subprocess.PIPE,
            close_fds=self._os != "windows",
        )

        time.sleep(float(self.cfg.boot_grace_seconds))

    def restart(self) -> "carla.Client":
        """Force restart the CARLA server and return a connected client."""
        self.shutdown(force=True)
        self.launch()
        return self._wait_until_ready()

    def shutdown(self, force: bool = False, wait_seconds: float = 10.0) -> None:
        """Stop CARLA server(s) this manager can find."""
        # If we spawned one, try to stop it first.
        if self.proc and self.proc.poll() is None:
            self._terminate_process(self.proc, force=force, wait_seconds=wait_seconds)

        # As a safety net, terminate any listener/known-name processes.
        for p in self._find_carla_processes():
            self._terminate_psutil_proc(p, force=force, wait_seconds=wait_seconds)

        self.proc = None

    def connect(self) -> "carla.Client":
        """Create a connected carla.Client to an already-running server."""
        if carla is None:
            raise RuntimeError("The CARLA Python API is not available. Install carla and try again.")
        client = carla.Client(self.cfg.host, int(self.cfg.port))
        client.set_timeout(float(self.cfg.timeout))
        _ = client.get_server_version()  # lightweight ping
        return client

    # Context manager support: ensure up & clean down (best-effort)
    def __enter__(self) -> "carla.Client":
        return self.ensure_running()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(force=False)

    # ---------- Internals ----------

    def _launcher_cmd(self) -> (str, List[str]):
        if "windows" in self._os:
            launcher = str(self.cfg.windows_launcher)
            if not os.path.isfile(launcher):
                raise FileNotFoundError(
                    f"CARLA launcher not found at {launcher}. "
                    f"Set windows_launcher in CARLA/config_yaml/default.yaml."
                )
            args = list(self.cfg.extra_args or [])
            if "windows" in self._os and "-opengl" in args:
                args.remove("-opengl")  # Unreal Windows build doesn’t support it
        elif "linux" in self._os:
            launcher = str(self.cfg.linux_launcher)
            if not os.path.isfile(launcher):
                raise FileNotFoundError(
                    f"CARLA launcher not found at {launcher}. "
                    f"Set linux_launcher in CARLA/config_yaml/default.yaml."
                )
            args = list(self.cfg.extra_args or [])
        else:
            raise RuntimeError(f"Unsupported OS: {self._os}")
        return launcher, args

    def _wait_until_ready(self) -> "carla.Client":
        if carla is None:
            raise RuntimeError("The CARLA Python API is not available. Install carla and try again.")

        deadline = time.time() + float(self.cfg.connect_retry_seconds)
        last_err: Optional[Exception] = None

        while time.time() < deadline:
            try:
                client = carla.Client(self.cfg.host, int(self.cfg.port))
                client.set_timeout(float(self.cfg.timeout))
                _ = client.get_server_version()
                return client
            except Exception as e:
                last_err = e
                # If the process we spawned died, relaunch once:
                if self.proc and self.proc.poll() is not None:
                    self.launch()
                time.sleep(10.0)

        raise RuntimeError(
            f"Could not connect to CARLA at {self.cfg.host}:{self.cfg.port} "
            f"within {self.cfg.connect_retry_seconds} seconds. Last error: {last_err}"
        )

    @staticmethod
    def _is_listening(port: int) -> bool:
        """Return True if any process is listening on the given TCP port."""
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
                return True
        return False

    def _find_carla_processes(self) -> List[psutil.Process]:
        """
        Find processes likely to be CARLA/Unreal server instances.
        Heuristics: process name and/or listening on configured port.
        """
        names = {"CarlaUE4", "CarlaUE4.exe", "UE4Editor", "UE4Editor.exe", "CarlaUE4.sh"}
        candidates: List[psutil.Process] = []

        # 1) Name-based scan
        for p in psutil.process_iter(attrs=["name"]):
            try:
                if p.info["name"] in names:
                    candidates.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 2) Port-based scan (listener on CARLA port)
        listeners = set()
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == int(self.cfg.port) and c.status == psutil.CONN_LISTEN and c.pid:
                listeners.add(c.pid)

        for pid in listeners:
            try:
                p = psutil.Process(pid)
                if p not in candidates:
                    candidates.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return candidates

    def _kill_stale_processes(self) -> None:
        """Terminate any existing CARLA server processes listening on the port."""
        for p in self._find_carla_processes():
            self._terminate_psutil_proc(p, force=True, wait_seconds=10.0)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen, force: bool, wait_seconds: float) -> None:
        try:
            if not force:
                if platform.system().lower().startswith("win"):
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=wait_seconds)
            else:
                if platform.system().lower().startswith("win"):
                    proc.kill()
                else:
                    proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=wait_seconds)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _terminate_psutil_proc(p: psutil.Process, force: bool, wait_seconds: float) -> None:
        try:
            if not force:
                p.terminate()
                p.wait(timeout=wait_seconds)
            else:
                p.kill()
                p.wait(timeout=wait_seconds)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied):
            pass
