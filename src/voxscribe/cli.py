#!/usr/bin/env python3
"""
Voxscribe CLI - Command-line interface for the speech-to-text daemon.

Usage:
    voxscribe setup      Install systemd service and create config
    voxscribe teardown   Remove systemd service
    voxscribe start      Start recording
    voxscribe stop       Stop recording
    voxscribe toggle     Toggle recording (default)
    voxscribe status     Check daemon status
"""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# Paths
SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "voxscribe.sock"
CONFIG_DIR = Path.home() / ".config" / "voxscribe"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_FILE = SYSTEMD_DIR / "voxscribe.service"
TIMEOUT = 15  # seconds


def get_service_content() -> str:
    """Generate systemd service file content with correct Python path."""
    python_path = sys.executable
    return f"""[Unit]
Description=Voxscribe - Real-time speech-to-text daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart={python_path} -m voxscribe.daemon
Restart=on-failure
RestartSec=3
Environment="XDG_RUNTIME_DIR=%t"
PassEnvironment=OPENAI_API_KEY
StandardOutput=journal
StandardError=journal
SyslogIdentifier=voxscribe

[Install]
WantedBy=default.target
"""


def get_default_config() -> str:
    """Return default configuration content."""
    return """# Voxscribe Configuration

# Logging level: debug, info, warning, error
log_level: info

# VAD (Voice Activity Detection) settings
vad:
  # Type: "server_vad" (volume-based) or "semantic_vad" (AI-based)
  type: server_vad

  # Sensitivity threshold (0.0-1.0, higher = more sensitive)
  threshold: 0.5

  # Audio to include BEFORE speech detected (ms)
  prefix_padding_ms: 300

  # Silence duration to detect end of speech (ms)
  # Lower = faster response, Higher = allows longer pauses
  silence_duration_ms: 1500

# Transcription settings
transcription:
  # Model: gpt-4o-transcribe (smart but edits speech), whisper-1 (literal)
  model: gpt-4o-transcribe

  # Prompt to guide transcription behavior
  prompt: "Transcribe exactly what is said, word for word. Include filler words, repetitions, false starts, and partial sentences. Do not edit, summarize, or clean up the speech in any way."

  # Language hint (ISO-639-1 code, e.g., "en", "de")
  # Leave empty for auto-detection
  language: ""
"""


def run_systemctl(*args: str) -> tuple[bool, str]:
    """Run systemctl command and return success status and output."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "systemctl not found"


def setup() -> int:
    """Install systemd service and create config directory."""
    print("Setting up voxscribe...")

    # Create config directory
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Config directory: {CONFIG_DIR}")

    # Create config file if not exists
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(get_default_config())
        print(f"  Created config: {CONFIG_FILE}")
    else:
        print(f"  Config exists: {CONFIG_FILE}")

    # Create systemd directory
    SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)

    # Write service file
    SERVICE_FILE.write_text(get_service_content())
    print(f"  Service file: {SERVICE_FILE}")

    # Reload systemd
    ok, _ = run_systemctl("daemon-reload")
    if not ok:
        print("  ERROR: Failed to reload systemd daemon")
        return 1
    print("  Reloaded systemd daemon")

    # Import environment variables for systemd
    subprocess.run(
        ["systemctl", "--user", "import-environment", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "OPENAI_API_KEY"],
        capture_output=True,
    )
    print("  Imported Wayland environment")

    # Enable service
    ok, _ = run_systemctl("enable", "voxscribe")
    if not ok:
        print("  ERROR: Failed to enable service")
        return 1
    print("  Enabled service")

    # Start service
    ok, output = run_systemctl("start", "voxscribe")
    if not ok:
        print(f"  ERROR: Failed to start service: {output}")
        return 1
    print("  Started service")

    # Verify socket
    import time
    for _ in range(10):
        if SOCKET_PATH.exists():
            break
        time.sleep(0.5)
    else:
        print(f"  WARNING: Socket not found at {SOCKET_PATH}")
        print("  Check logs: journalctl --user -u voxscribe -f")
        return 1

    print("\nSetup complete! Use 'voxscribe toggle' to start/stop recording.")
    print(f"Logs: journalctl --user -u voxscribe -f")
    return 0


def teardown() -> int:
    """Remove systemd service."""
    print("Removing voxscribe service...")

    # Stop service
    run_systemctl("stop", "voxscribe")
    print("  Stopped service")

    # Disable service
    run_systemctl("disable", "voxscribe")
    print("  Disabled service")

    # Remove service file
    if SERVICE_FILE.exists():
        SERVICE_FILE.unlink()
        print(f"  Removed {SERVICE_FILE}")

    # Reload systemd
    run_systemctl("daemon-reload")
    print("  Reloaded systemd daemon")

    print("\nTeardown complete!")
    print(f"Config preserved at: {CONFIG_DIR}")
    print("To fully uninstall: pipx uninstall voxscribe")
    return 0


def send_command(cmd: str) -> str:
    """Send command to daemon and return response."""
    if not SOCKET_PATH.exists():
        return "ERROR: Daemon not running (socket not found). Run 'voxscribe setup' first."

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


def main() -> NoReturn:
    """Entry point."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "toggle"
    cmd = cmd.lower()

    # Setup/teardown commands
    if cmd == "setup":
        sys.exit(setup())
    elif cmd == "teardown":
        sys.exit(teardown())

    # Daemon commands
    if cmd not in ("start", "stop", "status", "toggle"):
        print(__doc__)
        sys.exit(1)

    response = send_command(cmd)
    print(response)

    # Exit code based on response
    sys.exit(0 if response.startswith("OK") else 1)


if __name__ == "__main__":
    main()
