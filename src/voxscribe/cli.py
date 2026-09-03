#!/usr/bin/env python3
"""
Voxscribe CLI - Command-line interface for the speech-to-text daemon.

Usage:
    voxscribe setup             Install systemd service and create config
    voxscribe teardown          Remove systemd service
    voxscribe install-extension Install GNOME Shell extension
    voxscribe start             Start recording
    voxscribe stop              Stop recording
    voxscribe toggle            Toggle recording (default)
    voxscribe status            Check daemon status
    voxscribe transcribe [file]  Transcribe a PCM recording (default: latest)
"""

import asyncio
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import yaml

from voxscribe.clipboard import clipboard_payload, copy_text
from voxscribe.paths import RECORDINGS_DIR

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
PartOf=graphical-session.target
BindsTo=graphical-session.target

[Service]
Type=simple
ExecStart={python_path} -m voxscribe.daemon
Restart=always
RestartSec=3
Environment="XDG_RUNTIME_DIR=%t"
PassEnvironment=OPENAI_API_KEY ELEVENLABS_API_KEY
StandardOutput=journal
StandardError=journal
SyslogIdentifier=voxscribe

[Install]
WantedBy=graphical-session.target
"""


def get_default_config() -> str:
    """Return default configuration content."""
    return """# Voxscribe Configuration

# Logging level: debug, info, warning, error
log_level: info

# Transcription provider: openai or elevenlabs
provider: openai

# Language hint (ISO-639-1 code, e.g., "en", "de") - leave empty for auto-detection
language: ""

# Max seconds to wait for final transcription after stopping recording
transcription_timeout: 120

# OpenAI Realtime Transcription settings
openai:
  model: gpt-4o-transcribe
  prompt: "Transcribe exactly what is said, word for word. Include filler words, repetitions, false starts, and partial sentences. Do not edit, summarize, or clean up the speech in any way."
  vad_type: server_vad
  vad_threshold: 0.5
  vad_prefix_padding_ms: 300
  vad_silence_duration_ms: 1500

# ElevenLabs Scribe v2 Realtime settings
elevenlabs:
  vad_silence_threshold_secs: 1.5
  vad_threshold: 0.4
  enable_logging: false

# Client-side silence gate (only used with elevenlabs provider)
silence_gate:
  enabled: false
  threshold: 0.010
  gap_seconds: 3.0
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
        ["systemctl", "--user", "import-environment", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"],
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


def install_extension() -> int:
    """Install GNOME Shell extension."""
    import importlib.resources

    print("Installing GNOME Shell extension...")

    # Extension destination
    ext_dir = Path.home() / ".local" / "share" / "gnome-shell" / "extensions" / "voxscribe@frederikb.github.com"
    ext_dir.mkdir(parents=True, exist_ok=True)

    # Find the extension source directory
    # Try to find it relative to the package
    try:
        # When installed via pipx, use importlib.resources
        package_dir = Path(__file__).parent.parent.parent
        ext_src = package_dir / "extension"
        if not ext_src.exists():
            # Fallback: try finding it from cwd
            ext_src = Path.cwd() / "extension"
        if not ext_src.exists():
            print("  ERROR: Extension source not found")
            print("  Run this command from the voxscribe repository root")
            return 1
    except Exception as e:
        print(f"  ERROR: Could not locate extension: {e}")
        return 1

    # Copy extension files
    files_to_copy = ["extension.js", "metadata.json", "stylesheet.css", "prefs.js"]
    for filename in files_to_copy:
        src = ext_src / filename
        if src.exists():
            shutil.copy(src, ext_dir / filename)
            print(f"  Copied: {filename}")
        else:
            print(f"  WARNING: Missing {filename}")

    # Copy and compile schema
    schema_src = ext_src / "schemas"
    schema_dst = ext_dir / "schemas"
    if schema_src.exists():
        schema_dst.mkdir(exist_ok=True)
        for schema_file in schema_src.glob("*.xml"):
            shutil.copy(schema_file, schema_dst / schema_file.name)
            print(f"  Copied: schemas/{schema_file.name}")

        # Compile schemas
        try:
            result = subprocess.run(
                ["glib-compile-schemas", str(schema_dst)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  Compiled schemas")
            else:
                print(f"  ERROR: Schema compilation failed: {result.stderr}")
                return 1
        except FileNotFoundError:
            print("  ERROR: glib-compile-schemas not found")
            print("  Install: sudo apt install libglib2.0-dev-bin")
            return 1

    print(f"\nExtension installed to: {ext_dir}")
    print("\nTo activate:")
    print("  1. Log out and log back in (or restart GNOME Shell: Alt+F2, 'r', Enter on X11)")
    print("  2. Enable extension: gnome-extensions enable voxscribe@frederikb.github.com")
    print("  3. Open settings: gnome-extensions prefs voxscribe@frederikb.github.com")
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


def _create_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM data in a WAV header."""
    import struct

    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        len(pcm_data) + 36,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        len(pcm_data),
    )
    return header + pcm_data


def transcribe_file(file_path: str) -> int:
    """Transcribe a PCM audio file using batch API."""
    import json
    import urllib.error
    import urllib.request

    pcm_path = Path(file_path)
    if not pcm_path.exists():
        print(f"ERROR: File not found: {pcm_path}")
        return 1

    # Load config
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config not found: {CONFIG_FILE}")
        return 1
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    provider = config.get("provider", "openai")
    language = config.get("language", "")
    if language == "auto":
        language = ""

    # Get API key
    env_var = "OPENAI_API_KEY" if provider == "openai" else "ELEVENLABS_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"ERROR: {env_var} not set")
        return 1

    # Read PCM and wrap in WAV
    pcm_data = pcm_path.read_bytes()
    if not pcm_data:
        print("ERROR: Empty file")
        return 1

    wav_data = _create_wav(pcm_data)
    print(f"Transcribing {len(pcm_data)} bytes ({len(pcm_data) / 48000:.1f}s of audio)...")

    # Build multipart request
    boundary = "----VoxscribeBatch"
    body = b""

    if provider == "elevenlabs":
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers: dict[str, str] = {"xi-api-key": api_key}
        fields = {"model_id": "scribe_v2"}
        if language:
            fields["language_code"] = language
    else:
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        fields = {"model": "gpt-4o-transcribe"}
        if language:
            fields["language"] = language
        openai_config = config.get("openai", {})
        if openai_config.get("prompt"):
            fields["prompt"] = openai_config["prompt"]

    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()

    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="recording.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += wav_data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        response = urllib.request.urlopen(req, timeout=120)
        result = json.loads(response.read().decode())
        text = result.get("text", "")
        if text:
            print(f"\n{text}")
            try:
                method = asyncio.run(copy_text(clipboard_payload(text)))
                print(f"\nCopied {len(text)} chars to clipboard via {method}")
            except Exception as e:
                print(f"\nWARNING: clipboard delivery failed: {e}")
            return 0
        else:
            print("ERROR: Empty transcription result")
            return 1
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"ERROR: HTTP {e.code}: {error_body}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def main() -> NoReturn:
    """Entry point."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "toggle"
    cmd = cmd.lower()

    # Setup/teardown commands
    if cmd == "setup":
        sys.exit(setup())
    elif cmd == "teardown":
        sys.exit(teardown())
    elif cmd == "install-extension":
        sys.exit(install_extension())
    elif cmd == "transcribe":
        arg = sys.argv[2] if len(sys.argv) >= 3 else "latest"
        if arg == "latest":
            if not RECORDINGS_DIR.exists():
                print("ERROR: No recordings directory found")
                sys.exit(1)
            pcm_files = sorted(RECORDINGS_DIR.glob("rec-*.pcm"), key=lambda f: f.stat().st_mtime, reverse=True)
            if not pcm_files:
                print("ERROR: No recordings found")
                sys.exit(1)
            print(f"Using latest recording: {pcm_files[0].name}")
            sys.exit(transcribe_file(str(pcm_files[0])))
        else:
            sys.exit(transcribe_file(arg))

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
