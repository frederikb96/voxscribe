"""ElevenLabs credential lookup, shared by the daemon and the CLI."""

import os
import shlex
import subprocess
from typing import Any

ENV_VAR = "ELEVENLABS_API_KEY"
COMMAND_OPTION = "api_key_command"
COMMAND_TIMEOUT = 15  # seconds


class ApiKeyError(Exception):
    """The API key could not be resolved."""


def resolve_api_key(config: dict[str, Any]) -> str:
    """Return the ElevenLabs API key, raising ApiKeyError instead of returning an empty one.

    The config's api_key_command wins when set, so a key kept in a password store is
    fetched per process; otherwise the environment is used.
    """
    command = str(config.get(COMMAND_OPTION) or "").strip()
    if command:
        return _from_command(command)

    key = os.environ.get(ENV_VAR, "")
    if not key:
        raise ApiKeyError(f"{ENV_VAR} not set and no {COMMAND_OPTION} configured")
    return key


def _from_command(command: str) -> str:
    """Run command and return its stdout, which must be the key and nothing else."""
    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ApiKeyError(f"{COMMAND_OPTION} '{command}' failed: {e}") from e

    if result.returncode != 0:
        raise ApiKeyError(
            f"{COMMAND_OPTION} '{command}' exited {result.returncode}: {result.stderr.strip()}"
        )

    key = result.stdout.strip()
    if not key:
        raise ApiKeyError(f"{COMMAND_OPTION} '{command}' printed nothing")
    return key
