"""
vLLM HTTP Server Manager for automatic server lifecycle management.

This module provides utilities to start, monitor, and stop vLLM HTTP servers
for use with reward model inference in RL training pipelines.
"""

import atexit
import os
import signal
import subprocess
import time
from typing import Optional

import requests

from roll.utils.logging import get_logger

logger = get_logger()


class VLLMServerManager:
    """
    Manages the lifecycle of a vLLM HTTP server process.

    Usage:
        manager = VLLMServerManager(
            model_path="/path/to/model",
            gpu_id=3,
            port=8000,
        )
        url = manager.start()  # Returns "http://localhost:8000"
        # ... use the server ...
        manager.stop()  # Called automatically on exit
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        dtype: str = "auto",
        host: str = "0.0.0.0",
        startup_timeout: int = 300,
        log_file: Optional[str] = None,
    ):
        """
        Initialize the vLLM server manager.

        Args:
            model_path: Path to the model to serve
            gpu_id: GPU index to run the server on
            port: Port to listen on
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model context length
            dtype: Data type for model weights
            host: Host to bind to
            startup_timeout: Timeout in seconds to wait for server startup
            log_file: Path to log file for server output (default: /tmp/vllm_server_{port}.log)
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.host = host
        self.startup_timeout = startup_timeout
        self.log_file = log_file or f"/tmp/vllm_server_{port}.log"

        self.process: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None
        self._log_file_handle = None

        # Register cleanup on exit
        atexit.register(self.stop)

    def start(self) -> str:
        """
        Start the vLLM HTTP server.

        Returns:
            str: The server URL (e.g., "http://localhost:8000")

        Raises:
            RuntimeError: If server fails to start within timeout
        """
        if self.process is not None:
            logger.warning("vLLM server already running")
            return self.url

        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--host", self.host,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--trust-remote-code",
        ]

        # Set environment for specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        logger.info(f"Starting vLLM server on GPU {self.gpu_id}, port {self.port}")
        logger.info(f"Command: CUDA_VISIBLE_DEVICES={self.gpu_id} {' '.join(cmd)}")
        logger.info(f"Server log file: {self.log_file}")

        # Start process with output redirected to file (not pipe, to avoid buffer blocking)
        self._log_file_handle = open(self.log_file, "w")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file_handle,
            stderr=subprocess.STDOUT,
        )

        self.url = f"http://localhost:{self.port}"

        # Wait for server to be ready
        if not self._wait_for_ready():
            self.stop()
            raise RuntimeError(f"vLLM server failed to start within {self.startup_timeout}s")

        logger.info(f"vLLM server ready at {self.url}")
        return self.url

    def _wait_for_ready(self) -> bool:
        """
        Wait for the server to be ready by polling the health endpoint.

        Returns:
            bool: True if server is ready, False if timeout
        """
        health_url = f"{self.url}/health"
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            # Check if process died
            if self.process.poll() is not None:
                # Process exited, read log file for debugging
                logger.error(f"vLLM server process exited with code {self.process.returncode}")
                logger.error(f"Check server log at: {self.log_file}")
                try:
                    with open(self.log_file, "r") as f:
                        output = f.read()[-2000:]  # Last 2000 chars
                        logger.error(f"Server output (last 2000 chars):\n{output}")
                except Exception:
                    pass
                return False

            # Try health check
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            # Log progress
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0 and elapsed > 0:
                logger.info(f"Waiting for vLLM server... ({elapsed}s elapsed)")

            time.sleep(2)

        return False

    def stop(self):
        """Stop the vLLM server process."""
        if self.process is None:
            return

        logger.info("Stopping vLLM server...")

        # Try graceful termination first
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            logger.warning("vLLM server did not terminate gracefully, killing...")
            self.process.kill()
            self.process.wait()

        # Close log file handle
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

        self.process = None
        self.url = None
        logger.info(f"vLLM server stopped. Log file: {self.log_file}")

    def is_running(self) -> bool:
        """Check if the server process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


# Global server manager instance (singleton pattern for pipeline use)
_global_server_manager: Optional[VLLMServerManager] = None


def get_or_start_vllm_server(
    model_path: str,
    gpu_id: int,
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
) -> str:
    """
    Get or start a global vLLM server instance.

    This function ensures only one vLLM server is started per process,
    reusing the existing server if already running.

    Args:
        model_path: Path to the model to serve
        gpu_id: GPU index to run the server on
        port: Port to listen on
        gpu_memory_utilization: GPU memory utilization
        max_model_len: Maximum model context length

    Returns:
        str: The server URL
    """
    global _global_server_manager

    if _global_server_manager is not None and _global_server_manager.is_running():
        logger.info(f"Reusing existing vLLM server at {_global_server_manager.url}")
        return _global_server_manager.url

    _global_server_manager = VLLMServerManager(
        model_path=model_path,
        gpu_id=gpu_id,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    return _global_server_manager.start()


def stop_global_vllm_server():
    """Stop the global vLLM server if running."""
    global _global_server_manager

    if _global_server_manager is not None:
        _global_server_manager.stop()
        _global_server_manager = None
