#!/usr/bin/env python3
"""
Voxscribe Daemon - Streaming speech-to-text using OpenAI Realtime API.

Runs as a systemd user service, listens on Unix socket for commands.
Commands: START, STOP, STATUS, TOGGLE

Emits DBus signals for GNOME extension integration.
"""

import asyncio
import base64
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import websockets
import yaml

try:
    from dbus_next.aio import MessageBus
    from dbus_next.service import ServiceInterface, method, signal as dbus_signal

    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

# Configuration
SAMPLE_RATE = 24000
CHUNK_BYTES = 4800  # 100ms of audio at 24kHz 16-bit mono
SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "voxscribe.sock"
CONFIG_FILE = Path.home() / ".config" / "voxscribe" / "config.yaml"
OUTPUT_DIR = Path("/tmp")
RESULT_SYMLINK = OUTPUT_DIR / "voxscribe-result.txt"

# Sound files
SOUND_START = Path("/usr/share/sounds/freedesktop/stereo/device-added.oga")
SOUND_STOP = Path("/usr/share/sounds/freedesktop/stereo/message.oga")
SOUND_DONE = Path("/usr/share/sounds/freedesktop/stereo/complete.oga")

# DBus configuration
DBUS_NAME = "com.github.frederikb.Voxscribe"
DBUS_PATH = "/com/github/frederikb/Voxscribe"

# Logger setup
logger = logging.getLogger("voxscribe")


def setup_logging(level: str) -> None:
    """Configure logging with specified level."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    log_level = level_map.get(level.lower(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.info(f"Log level: {level.upper()}")


def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml."""
    if not CONFIG_FILE.exists():
        logger.warning(f"Config not found: {CONFIG_FILE}, using defaults")
        return {
            "log_level": "info",
            "vad": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 1500,
            },
            "transcription": {
                "model": "gpt-4o-transcribe",
                "prompt": "",
                "language": "",
            },
        }

    try:
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {CONFIG_FILE}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


class State(Enum):
    """Daemon state machine."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


# DBus interface for GNOME extension
if DBUS_AVAILABLE:

    class VoxscribeDBusInterface(ServiceInterface):
        """DBus interface for status updates to GNOME extension."""

        def __init__(self) -> None:
            super().__init__(DBUS_NAME)
            self._state = "idle"
            self._text = ""

        @dbus_signal()
        def StateChanged(self) -> "ss":
            """Signal emitted when state changes. Returns (state, text)."""
            return [self._state, self._text]

        @method()
        def GetStatus(self) -> "ss":
            """Get current state and last transcription text."""
            return [self._state, self._text]

        def emit_state(self, state: str, text: str = "") -> None:
            """Emit state change signal."""
            self._state = state
            self._text = text
            self.StateChanged()


class VoxscribeDaemon:
    """Main daemon class managing recording and transcription."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize daemon with configuration."""
        self.state = State.IDLE
        self.api_key: str = ""
        self.config = config
        self.transcripts: dict[str, str] = {}
        self.pending_items: set[str] = set()
        self.pw_record_proc: Optional[asyncio.subprocess.Process] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.recording_task: Optional[asyncio.Task[None]] = None
        self.shutdown_event = asyncio.Event()
        self.current_output_file: Optional[Path] = None
        self.dbus_interface: Optional[Any] = None
        self.dbus_bus: Optional[Any] = None

    async def setup_dbus(self) -> None:
        """Set up DBus service for GNOME extension communication."""
        if not DBUS_AVAILABLE:
            logger.info("DBus not available (dbus-next not installed)")
            return

        try:
            self.dbus_bus = await MessageBus().connect()
            self.dbus_interface = VoxscribeDBusInterface()
            self.dbus_bus.export(DBUS_PATH, self.dbus_interface)
            await self.dbus_bus.request_name(DBUS_NAME)
            logger.info(f"DBus service registered: {DBUS_NAME}")
        except Exception as e:
            logger.warning(f"DBus setup failed (extension won't work): {e}")
            self.dbus_interface = None

    def emit_state(self, state: str, text: str = "") -> None:
        """Emit state change via DBus."""
        if self.dbus_interface:
            self.dbus_interface.emit_state(state, text[-50:] if text else "")

    def load_api_key(self) -> bool:
        """Load OpenAI API key from environment variable."""
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.api_key:
            logger.info("API key loaded from environment")
            return True
        logger.error("OPENAI_API_KEY environment variable not set")
        return False

    def play_sound(self, sound_file: Path) -> None:
        """Play sound file asynchronously (non-blocking)."""
        if sound_file.exists():
            try:
                subprocess.Popen(
                    ["pw-play", str(sound_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.debug(f"Sound play failed: {e}")

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to clipboard using wl-copy."""
        try:
            subprocess.Popen(
                ["wl-copy", "--", text],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info(f"Copied {len(text)} chars to clipboard")
            return True
        except FileNotFoundError:
            logger.error("wl-copy not found")
            return False
        except Exception as e:
            logger.error(f"Clipboard copy failed: {e}")
            return False

    async def start_recording(self) -> tuple[bool, str]:
        """Start recording and transcription session."""
        if self.state != State.IDLE:
            return False, f"Cannot start: state is {self.state.value}"

        # Set state immediately to prevent race conditions from rapid toggles
        self.state = State.RECORDING
        self.play_sound(SOUND_START)
        self.emit_state("recording", "")
        logger.info("Starting recording session")
        self.transcripts = {}
        self.pending_items = set()

        # Create timestamped output file and update symlink
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_output_file = OUTPUT_DIR / f"voxscribe-{timestamp}.txt"
        self.current_output_file.touch()
        RESULT_SYMLINK.unlink(missing_ok=True)
        RESULT_SYMLINK.symlink_to(self.current_output_file)
        logger.info(f"Output file: {self.current_output_file.name}")

        # Start pw-record
        try:
            self.pw_record_proc = await asyncio.create_subprocess_exec(
                "pw-record",
                "--rate",
                str(SAMPLE_RATE),
                "--format",
                "s16",
                "--channels",
                "1",
                "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info(f"pw-record started (PID {self.pw_record_proc.pid})")
        except Exception as e:
            logger.error(f"Failed to start pw-record: {e}")
            self.state = State.IDLE
            self.emit_state("idle", "")
            return False, f"Failed to start audio capture: {e}"

        # Connect to OpenAI
        try:
            url = "wss://api.openai.com/v1/realtime?intent=transcription"
            headers = {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}
            self.websocket = await asyncio.wait_for(
                websockets.connect(url, additional_headers=headers, max_size=None),
                timeout=10,
            )
            logger.info("WebSocket connected")

            # Wait for session.created
            msg = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            ev = json.loads(msg)
            logger.debug(f"Received: {ev.get('type')}")

            # Configure session
            vad_config = self.config["vad"]
            transcription_config = self.config["transcription"]

            transcription_settings: dict[str, Any] = {"model": transcription_config["model"]}
            if transcription_config.get("prompt"):
                transcription_settings["prompt"] = transcription_config["prompt"]
            if transcription_config.get("language"):
                transcription_settings["language"] = transcription_config["language"]

            session_config: dict[str, Any] = {
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription_settings,
                "turn_detection": {
                    "type": vad_config["type"],
                    "threshold": vad_config["threshold"],
                    "prefix_padding_ms": vad_config["prefix_padding_ms"],
                    "silence_duration_ms": vad_config["silence_duration_ms"],
                },
            }

            logger.info(f"VAD: {vad_config['type']}, silence={vad_config['silence_duration_ms']}ms")
            if transcription_config.get("prompt"):
                logger.debug(f"Prompt: {transcription_config['prompt'][:80]}...")

            await self.websocket.send(
                json.dumps({"type": "transcription_session.update", "session": session_config})
            )
            msg = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            ev = json.loads(msg)
            logger.debug(f"Session configured: {ev.get('type')}")

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            if self.pw_record_proc:
                self.pw_record_proc.terminate()
                self.pw_record_proc = None
            self.state = State.IDLE
            self.emit_state("idle", "")
            return False, f"Failed to connect to OpenAI: {e}"

        # Start recording tasks
        self.recording_task = asyncio.create_task(self._recording_loop())
        logger.info("Recording started")
        return True, "Recording started"

    async def stop_recording(self) -> tuple[bool, str]:
        """Stop recording and wait for final transcription."""
        if self.state == State.IDLE:
            return False, "Not recording"
        if self.state == State.TRANSCRIBING:
            return False, "Already stopping, please wait"

        logger.info("Stopping recording")
        self.play_sound(SOUND_STOP)
        self.state = State.TRANSCRIBING
        self.emit_state("transcribing", self._get_current_text())

        # Cancel recording task first to free websocket recv
        if self.recording_task and not self.recording_task.done():
            self.recording_task.cancel()
            try:
                await self.recording_task
            except asyncio.CancelledError:
                pass
            logger.debug("Recording task cancelled")

        # Terminate pw-record
        if self.pw_record_proc:
            self.pw_record_proc.terminate()
            try:
                await asyncio.wait_for(self.pw_record_proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.pw_record_proc.kill()
            logger.info("pw-record stopped")
            self.pw_record_proc = None

        # Send commit to force transcription
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                logger.info("Sent audio commit")
            except Exception as e:
                logger.error(f"Failed to send commit: {e}")

        # Wait for transcriptions (event-driven)
        wait_start = asyncio.get_event_loop().time()
        safety_timeout = 30

        while self.pending_items and (asyncio.get_event_loop().time() - wait_start) < safety_timeout:
            if not self.websocket:
                break

            try:
                msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                self._handle_event(json.loads(msg))
            except asyncio.TimeoutError:
                logger.debug(f"Waiting for {len(self.pending_items)} pending transcriptions...")
                continue
            except websockets.ConnectionClosed:
                logger.warning("WebSocket closed while waiting")
                break
            except Exception as e:
                logger.error(f"Event recv error: {e}")
                break

        if self.pending_items:
            logger.warning(f"Exiting with {len(self.pending_items)} pending items")
        else:
            logger.info("All transcriptions complete")

        # Close websocket
        if self.websocket:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=1)
            except Exception:
                pass
            self.websocket = None
            logger.info("WebSocket closed")

        # Process result
        result = " ".join(self.transcripts[k] for k in sorted(self.transcripts)).strip()
        logger.info(f"Final transcription: {len(result)} chars")

        if result:
            # Final write to timestamped file
            if self.current_output_file:
                self.current_output_file.write_text(result)
                logger.info(f"Saved: {self.current_output_file.name}")

            clipboard_text = f"stt-rec: {result}"
            self.copy_to_clipboard(clipboard_text)

        self.play_sound(SOUND_DONE)
        self.emit_state("done", result[-50:] if result else "")
        self.state = State.IDLE

        # Schedule idle state emission after 5 seconds
        asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))

        return True, f"Transcription complete: {len(result)} chars"

    def _get_current_text(self) -> str:
        """Get current transcription text."""
        return " ".join(self.transcripts[k] for k in sorted(self.transcripts)).strip()

    def _write_result_file(self) -> None:
        """Write current transcripts to output file (live updates)."""
        if not self.current_output_file:
            return
        result = self._get_current_text()
        if result:
            self.current_output_file.write_text(result)

    def _handle_event(self, ev: dict[str, Any]) -> None:
        """Handle OpenAI event."""
        t = ev.get("type", "")
        item_id = ev.get("item_id", "")

        if t == "input_audio_buffer.speech_started":
            if item_id:
                self.pending_items.add(item_id)
                logger.debug(f"Speech started: {item_id}")

        elif t == "conversation.item.input_audio_transcription.delta":
            delta = ev.get("delta", "")
            if item_id and delta:
                self.transcripts[item_id] = self.transcripts.get(item_id, "") + delta
                logger.debug(f"Delta [{item_id[:8]}]: +{len(delta)} chars")
                self._write_result_file()
                self.emit_state("recording", self._get_current_text())

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = ev.get("transcript", "")
            if item_id:
                if transcript:
                    self.transcripts[item_id] = transcript
                self.pending_items.discard(item_id)
                logger.info(f"Transcription completed [{item_id[:8]}]: {len(transcript)} chars")
                self._write_result_file()
                self.emit_state("recording", self._get_current_text())

        elif t == "input_audio_buffer.committed":
            committed_item_id = ev.get("item_id", "")
            if committed_item_id:
                self.pending_items.add(committed_item_id)
                logger.info(f"Commit created item: {committed_item_id[:8]}")

        elif t == "error":
            logger.error(f"API error: {ev.get('error', {})}")

    async def _recording_loop(self) -> None:
        """Main loop for sending audio and receiving events."""

        async def send_audio() -> None:
            while self.state == State.RECORDING and self.pw_record_proc and self.websocket:
                try:
                    assert self.pw_record_proc.stdout is not None
                    chunk = await asyncio.wait_for(
                        self.pw_record_proc.stdout.read(CHUNK_BYTES),
                        timeout=0.2,
                    )
                    if not chunk:
                        break
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode(),
                            }
                        )
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Send audio error: {e}")
                    break

        async def recv_events() -> None:
            while self.state == State.RECORDING and self.websocket:
                try:
                    msg = await asyncio.wait_for(self.websocket.recv(), timeout=0.2)
                    self._handle_event(json.loads(msg))
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Recv event error: {e}")
                    break

        await asyncio.gather(send_audio(), recv_events(), return_exceptions=True)

    async def handle_command(self, cmd: str) -> str:
        """Handle incoming command from client."""
        cmd = cmd.strip().upper()
        logger.info(f"Received command: {cmd}")

        if cmd == "START":
            ok, msg = await self.start_recording()
            return f"{'OK' if ok else 'ERROR'}: {msg}"
        elif cmd == "STOP":
            ok, msg = await self.stop_recording()
            return f"{'OK' if ok else 'ERROR'}: {msg}"
        elif cmd == "STATUS":
            return f"OK: {self.state.value}"
        elif cmd == "TOGGLE":
            if self.state == State.IDLE:
                ok, msg = await self.start_recording()
            else:
                ok, msg = await self.stop_recording()
            return f"{'OK' if ok else 'ERROR'}: {msg}"
        else:
            return f"ERROR: Unknown command '{cmd}'"

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection."""
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=5)
            if data:
                cmd = data.decode().strip()
                response = await self.handle_command(cmd)
                writer.write(f"{response}\n".encode())
                await writer.drain()
        except asyncio.TimeoutError:
            writer.write(b"ERROR: Timeout\n")
            await writer.drain()
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Cleaning up...")

        if self.state != State.IDLE:
            await self.stop_recording()

        if self.pw_record_proc:
            self.pw_record_proc.kill()

        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass

        if self.dbus_bus:
            try:
                self.dbus_bus.disconnect()
            except Exception:
                pass

        try:
            SOCKET_PATH.unlink(missing_ok=True)
        except Exception:
            pass

        logger.info("Cleanup complete")

    async def run(self) -> None:
        """Main daemon loop."""
        if not self.load_api_key():
            sys.exit(1)

        SOCKET_PATH.unlink(missing_ok=True)

        # Set up DBus for GNOME extension
        await self.setup_dbus()

        loop = asyncio.get_event_loop()

        def signal_handler() -> None:
            logger.info("Received shutdown signal")
            self.shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        server = await asyncio.start_unix_server(self.handle_client, path=str(SOCKET_PATH))
        SOCKET_PATH.chmod(0o600)
        logger.info(f"Listening on {SOCKET_PATH}")

        await self.shutdown_event.wait()

        server.close()
        await server.wait_closed()
        await self.cleanup()


def main() -> None:
    """Entry point."""
    print("[DAEMON] Voxscribe daemon starting...", flush=True)

    config = load_config()
    setup_logging(config.get("log_level", "info"))

    daemon = VoxscribeDaemon(config)
    asyncio.run(daemon.run())

    print("[DAEMON] Voxscribe daemon stopped", flush=True)


if __name__ == "__main__":
    main()
