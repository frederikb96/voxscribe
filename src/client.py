#!/usr/bin/env python3
"""
Long STT Client - Send commands to the daemon.

Usage: long-stt-ctl [start|stop|status|toggle]
Default command is 'toggle' if none specified.
"""

import os
import socket
import sys
from pathlib import Path

SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "long-stt.sock"
TIMEOUT = 15  # seconds


def send_command(cmd: str) -> str:
    """Send command to daemon and return response."""
    if not SOCKET_PATH.exists():
        return "ERROR: Daemon not running (socket not found)"

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        sock.connect(str(SOCKET_PATH))
        sock.sendall(f"{cmd}\n".encode())
        response = sock.recv(4096).decode().strip()
        sock.close()
        return response
    except socket.timeout:
        return "ERROR: Daemon timeout"
    except ConnectionRefusedError:
        return "ERROR: Daemon not accepting connections"
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    """Entry point."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "toggle"
    cmd = cmd.lower()

    if cmd not in ("start", "stop", "status", "toggle"):
        print(f"Usage: {sys.argv[0]} [start|stop|status|toggle]")
        sys.exit(1)

    response = send_command(cmd)
    print(response)

    # Exit code based on response
    sys.exit(0 if response.startswith("OK") else 1)


if __name__ == "__main__":
    main()
