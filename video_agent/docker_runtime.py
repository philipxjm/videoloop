"""
Docker Runtime for Video Understanding Agent
Manages the sandbox container for video processing
"""

import os
import time
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import docker
from docker.models.containers import Container

from .logging_utils import get_logger
from .transcript_cache import TranscriptCache

logger = get_logger(__name__)

_transcript_cache = TranscriptCache()


class DockerRuntime:
    """
    Manages a Docker container for video processing operations.
    """

    def __init__(
        self,
        image: str = "video-understanding-sandbox:latest",
        gpu: bool = True,
        memory_limit: str = "32g",
        shm_size: str = "8g",
        volumes: Optional[List[Dict[str, str]]] = None,
        environment: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ):
        self.image = image
        self.gpu = gpu
        self.memory_limit = memory_limit
        self.shm_size = shm_size
        self.volumes = volumes or []
        self.environment = environment or {}
        self.timeout = timeout
        
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        
        logger.info(f"DockerRuntime initialized with image: {image}")

    def start(self) -> None:
        """Start the container."""
        if self.container is not None:
            logger.warning("Container already running")
            return
        
        # Build volume mounts
        volume_mounts = {}
        for vol in self.volumes:
            source = os.path.abspath(vol["source"])
            target = vol["target"]
            mode = "ro" if vol.get("read_only", False) else "rw"
            volume_mounts[source] = {"bind": target, "mode": mode}
        
        # GPU configuration
        device_requests = []
        if self.gpu:
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        
        logger.info(f"Starting container: {self.image}")
        
        self.container = self.client.containers.run(
            self.image,
            command="sleep infinity",
            detach=True,
            remove=True,
            mem_limit=self.memory_limit,
            shm_size=self.shm_size,
            volumes=volume_mounts,
            environment=self.environment,
            device_requests=device_requests,
            working_dir="/testbed",
        )
        
        # Wait for container to be ready
        time.sleep(1)
        logger.info(f"Container started: {self.container.short_id}")

    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container is not None:
            logger.info(f"Stopping container: {self.container.short_id}")
            try:
                self.container.stop(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            self.container = None

    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str]:
        """
        Execute a command in the container.
        
        Returns:
            Tuple of (exit_code, output)
        """
        if self.container is None:
            raise RuntimeError("Container not running. Call start() first.")
        
        timeout = timeout or self.timeout
        
        try:
            # Use --norc to skip /etc/bash.bashrc which prints CUDA banner to stdout
            exec_result = self.container.exec_run(
                cmd=["bash", "--norc", "-c", command],
                demux=True,
                workdir="/testbed",
            )
            
            exit_code = exec_result.exit_code
            stdout = exec_result.output[0] or b""
            stderr = exec_result.output[1] or b""
            
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")
            
            return exit_code, output.strip()
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return 1, str(e)

    def copy_to_container(self, local_path: str, container_path: str) -> bool:
        """Copy a file or directory to the container."""
        if self.container is None:
            raise RuntimeError("Container not running")
        
        try:
            # Create tar archive
            tar_stream = BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(local_path, arcname=os.path.basename(container_path))
            tar_stream.seek(0)
            
            # Copy to container
            container_dir = os.path.dirname(container_path)
            self.container.put_archive(container_dir, tar_stream)
            return True
            
        except Exception as e:
            logger.error(f"Error copying to container: {e}")
            return False

    def copy_from_container(self, container_path: str, local_path: str) -> bool:
        """Copy a file or directory from the container."""
        if self.container is None:
            raise RuntimeError("Container not running")
        
        try:
            # Get archive from container
            bits, stat = self.container.get_archive(container_path)
            
            # Extract to local path
            tar_stream = BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                tar.extractall(path=os.path.dirname(local_path))
            
            return True
            
        except Exception as e:
            logger.error(f"Error copying from container: {e}")
            return False

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata."""
        import json
        cmd = f"python /usr/local/bin/video_info.py {video_path}"
        exit_code, output = self.execute(cmd)
        
        if exit_code != 0:
            return {"error": output}
        
        # Strip stderr portion if present
        if "\n[STDERR]" in output:
            output = output.split("\n[STDERR]")[0]
        
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"error": "Failed to parse video info", "raw": output}

    def transcribe_audio(
        self,
        video_path: str,
        model_size: str = "base",
        context_limit_chars: int = 128000,
    ) -> Dict[str, Any]:
        """Transcribe audio from video, with multi-layer caching.

        Lookup order (handled by TranscriptCache):
        1. Global in-memory cache
        2. Segments-only cache (dataset/transcripts/{videoID}.json)
        3. Legacy cache (videomme-sampled/transcripts_large/{videoName}.large.json)
        4. Run Whisper in container as fallback
        """
        def _whisper_fallback(vpath: str):
            cmd = f"python /usr/local/bin/transcribe.py {vpath} large"
            return self.execute(cmd, timeout=600)

        result = _transcript_cache.get(video_path, whisper_fallback=_whisper_fallback)

        # Write transcript to a file in the container so the agent can grep/read it
        transcript_text = result.get("transcript", "")
        if transcript_text and self.container is not None:
            video_name = os.path.basename(video_path)
            video_id = video_name.rsplit(".", 1)[0] if "." in video_name else video_name
            transcript_file = f"/tmp/transcripts/{video_id}.txt"
            write_cmd = f"mkdir -p /tmp/transcripts && cat > {transcript_file} << 'TRANSCRIPT_EOF'\n{transcript_text}\nTRANSCRIPT_EOF"
            try:
                self.execute(write_cmd, timeout=10)
                logger.info(f"Wrote transcript to {transcript_file}")
            except Exception as e:
                logger.warning(f"Failed to write transcript file: {e}")

        return result

    @staticmethod
    def clear_transcript_cache() -> None:
        """Clear the global transcript cache."""
        _transcript_cache.clear()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
