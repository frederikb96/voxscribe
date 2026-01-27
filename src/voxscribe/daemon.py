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
SOUND_ERROR = Path("/usr/share/sounds/freedesktop/stereo/dialog-warning.oga")

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
    """Load configuration from config.yaml. Fails if not found."""
    if not CONFIG_FILE.exists():
        print(f"[CONFIG] ERROR: Config not found: {CONFIG_FILE}", flush=True)
        print("[CONFIG] Run 'voxscribe setup' first to create config.", flush=True)
        sys.exit(1)

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
        self.failed_items: set[str] = set()
        self.item_creation_times: dict[str, float] = {}
        self.pw_record_proc: Optional[asyncio.subprocess.Process] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.recording_task: Optional[asyncio.Task[None]] = None
        self.shutdown_event = asyncio.Event()
        self.force_stop_event = asyncio.Event()
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
            self.dbus_interface.emit_state(state, text)

    def load_api_key(self) -> bool:
        """Load OpenAI API key from environment variable."""
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.api_key:
            logger.info("API key loaded from environment")
            return True
        logger.error("OPENAI_API_KEY environment variable not set")
        return False

    async def _terminate_pw_record(self) -> None:
        """Properly terminate pw-record process with wait and kill fallback."""
        if not self.pw_record_proc:
            return
        try:
            self.pw_record_proc.terminate()
            await asyncio.wait_for(self.pw_record_proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            logger.warning("pw-record didn't respond to SIGTERM, sending SIGKILL")
            self.pw_record_proc.kill()
            try:
                await asyncio.wait_for(self.pw_record_proc.wait(), timeout=1)
            except asyncio.TimeoutError:
                logger.error("pw-record didn't respond to SIGKILL!")
        except Exception as e:
            logger.error(f"Error terminating pw-record: {e}")
        finally:
            self.pw_record_proc = None
        logger.info("pw-record stopped")

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
        self.failed_items = set()
        self.item_creation_times = {}

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
            self.play_sound(SOUND_ERROR)
            self.emit_state("error", "")
            self.state = State.IDLE
            asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))
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

            # Configure session - only include values that are set in config
            vad_config = self.config.get("vad", {})
            transcription_config = self.config.get("transcription", {})

            # Transcription settings - model is required
            if "model" not in transcription_config:
                raise ValueError("transcription.model is required in config")
            transcription_settings: dict[str, Any] = {"model": transcription_config["model"]}
            if transcription_config.get("prompt"):
                transcription_settings["prompt"] = transcription_config["prompt"]
            if transcription_config.get("language"):
                transcription_settings["language"] = transcription_config["language"]

            # VAD settings - only include values that are set, let OpenAI use defaults
            turn_detection: dict[str, Any] = {"type": vad_config.get("type", "server_vad")}
            if "threshold" in vad_config:
                turn_detection["threshold"] = vad_config["threshold"]
            if "prefix_padding_ms" in vad_config:
                turn_detection["prefix_padding_ms"] = vad_config["prefix_padding_ms"]
            if "silence_duration_ms" in vad_config:
                turn_detection["silence_duration_ms"] = vad_config["silence_duration_ms"]

            session_config: dict[str, Any] = {
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription_settings,
                "turn_detection": turn_detection,
            }

            logger.info(f"VAD: {turn_detection.get('type')}, config keys: {list(turn_detection.keys())}")
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
            await self._terminate_pw_record()
            self.play_sound(SOUND_ERROR)
            self.emit_state("error", "")
            self.state = State.IDLE
            asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))
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
            # Force stop: interrupt the transcription wait loop
            logger.info("Force stop requested - aborting transcription wait")
            self.force_stop_event.set()
            return True, "Force stopping..."

        # Clear force stop for normal stop flow
        self.force_stop_event.clear()
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
        await self._terminate_pw_record()

        # Send commit to force transcription
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                logger.info("Sent audio commit")
            except Exception as e:
                logger.error(f"Failed to send commit: {e}")

        # Wait for transcriptions (event-driven, interruptible by force_stop)
        wait_start = asyncio.get_event_loop().time()
        safety_timeout = self.config.get("transcription_timeout", 120)
        wait_exit_reason = "completed"

        while self.pending_items and (asyncio.get_event_loop().time() - wait_start) < safety_timeout:
            if self.force_stop_event.is_set():
                wait_exit_reason = "force_stopped"
                break
            if not self.websocket:
                wait_exit_reason = "websocket_closed"
                break

            try:
                msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                self._handle_event(json.loads(msg))
            except asyncio.TimeoutError:
                logger.debug(f"Waiting for {len(self.pending_items)} pending transcriptions...")
                continue
            except websockets.ConnectionClosed:
                logger.warning("WebSocket closed while waiting")
                wait_exit_reason = "websocket_closed"
                break
            except Exception as e:
                logger.error(f"Event recv error: {e}")
                wait_exit_reason = "error"
                break

        # Check if we exited due to timeout
        if self.pending_items and wait_exit_reason == "completed":
            wait_exit_reason = "timeout"

        # Log exit reason
        if wait_exit_reason == "completed":
            logger.info("All transcriptions complete")
        elif wait_exit_reason == "force_stopped":
            logger.info(f"Wait aborted by user ({len(self.pending_items)} items pending)")
        elif wait_exit_reason == "timeout":
            logger.warning(f"Timeout: exiting with {len(self.pending_items)} pending items")
        else:
            logger.warning(f"Wait ended ({wait_exit_reason}) with {len(self.pending_items)} pending")

        # Close websocket
        if self.websocket:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=1)
            except Exception:
                pass
            self.websocket = None
            logger.info("WebSocket closed")

        # Process result - always save whatever we have
        result = " ".join(self.transcripts[k] for k in sorted(self.transcripts)).strip()
        logger.info(f"Final transcription: {len(result)} chars")

        if result:
            if self.current_output_file:
                self.current_output_file.write_text(result)
                logger.info(f"Saved: {self.current_output_file.name}")

            clipboard_text = f"stt-rec: {result}"
            self.copy_to_clipboard(clipboard_text)

        # Determine outcome and play appropriate sound
        has_failures = bool(self.failed_items or self.pending_items)
        if has_failures:
            lost_count = len(self.failed_items) + len(self.pending_items)
            logger.warning(f"Partial transcription: {lost_count} items failed/pending")
            self.play_sound(SOUND_ERROR)
            self.emit_state("partial", result[-50:] if result else "")
        else:
            self.play_sound(SOUND_DONE)
            self.emit_state("done", result[-50:] if result else "")

        self.state = State.IDLE

        # Schedule idle state emission after 5 seconds
        asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))

        status = "partial" if has_failures else "complete"
        return True, f"Transcription {status}: {len(result)} chars"

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

        elif t == "conversation.item.input_audio_transcription.delta":
            delta = ev.get("delta", "")
            if item_id and delta:
                self.transcripts[item_id] = self.transcripts.get(item_id, "") + delta
                self._write_result_file()
                self.emit_state("recording", self._get_current_text())

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = ev.get("transcript", "")
            if item_id:
                if transcript:
                    self.transcripts[item_id] = transcript
                self.pending_items.discard(item_id)
                # Calculate and log latency
                created_at = self.item_creation_times.pop(item_id, None)
                if created_at:
                    latency = asyncio.get_event_loop().time() - created_at
                    logger.info(f"Transcription completed [{item_id[:8]}]: {len(transcript)} chars ({latency:.1f}s latency)")
                else:
                    logger.info(f"Transcription completed [{item_id[:8]}]: {len(transcript)} chars")
                self._write_result_file()
                self.emit_state("recording", self._get_current_text())

        elif t == "input_audio_buffer.committed":
            committed_item_id = ev.get("item_id", "")
            if committed_item_id:
                self.pending_items.add(committed_item_id)
                self.item_creation_times[committed_item_id] = asyncio.get_event_loop().time()
                logger.info(f"Commit created item: {committed_item_id[:8]}")

        elif t == "conversation.item.input_audio_transcription.failed":
            error = ev.get("error", {})
            if item_id:
                self.failed_items.add(item_id)
                self.pending_items.discard(item_id)
                self.item_creation_times.pop(item_id, None)
                logger.error(
                    f"Transcription failed [{item_id[:8]}]: "
                    f"{error.get('code', 'unknown')} - {error.get('message', 'No message')}"
                )

        elif t == "error":
            logger.error(f"API error: {ev.get('error', {})}")

    async def _handle_recording_failure(self) -> None:
        """Handle WebSocket failure during recording - save partial data and notify user."""
        logger.warning("Handling recording failure - saving partial data")

        # Stop pw-record
        await self._terminate_pw_record()

        # Close websocket if still open
        if self.websocket:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=1)
            except Exception:
                pass
            self.websocket = None

        # Save whatever transcripts we have
        result = self._get_current_text()
        if result:
            if self.current_output_file:
                self.current_output_file.write_text(result)
                logger.info(f"Saved partial: {self.current_output_file.name}")
            clipboard_text = f"stt-rec: {result}"
            self.copy_to_clipboard(clipboard_text)
            logger.info(f"Saved {len(result)} chars before connection loss")

        # Notify user of failure
        self.play_sound(SOUND_ERROR)
        self.emit_state("partial", result[-50:] if result else "Connection lost")
        self.state = State.IDLE
        asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))

    async def _alert_transcription_stall(self, item_id: str, age: float) -> None:
        """Alert user that transcription is stalled during recording."""
        logger.warning(f"Transcription stalled for {item_id[:8]} ({age:.1f}s with no completion)")
        self.play_sound(SOUND_ERROR)
        self.emit_state("partial", f"Transcription delayed ({age:.0f}s)")

    async def _recording_loop(self) -> None:
        """Main loop for sending audio and receiving events.

        Uses asyncio.wait(FIRST_COMPLETED) so we react immediately when
        either WebSocket task exits, rather than waiting for all tasks.
        """
        logger.info("Recording loop started")
        loop_start = asyncio.get_event_loop().time()
        websocket_error = False
        exit_reason = "unknown"

        async def send_audio() -> None:
            """Send audio chunks to WebSocket."""
            nonlocal websocket_error, exit_reason
            logger.debug("send_audio task started")
            try:
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
                        raise
                    except Exception as e:
                        logger.error(f"Send audio error: {e}")
                        websocket_error = True
                        exit_reason = "send_audio_error"
                        break
            finally:
                logger.debug("send_audio task exiting")

        async def recv_events() -> None:
            """Receive and handle WebSocket events."""
            nonlocal websocket_error, exit_reason
            logger.debug("recv_events task started")
            try:
                while self.state == State.RECORDING and self.websocket:
                    try:
                        msg = await asyncio.wait_for(self.websocket.recv(), timeout=0.2)
                        self._handle_event(json.loads(msg))
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Recv event error: {e}")
                        websocket_error = True
                        exit_reason = "recv_events_error"
                        break
            finally:
                logger.debug("recv_events task exiting")

        async def monitor_transcriptions() -> None:
            """Monitor pending transcriptions and alert if stalled."""
            logger.debug("monitor task started")
            alerted_items: set[str] = set()
            heartbeat_interval = 30
            last_heartbeat = asyncio.get_event_loop().time()
            try:
                while self.state == State.RECORDING:
                    await asyncio.sleep(5)

                    now = asyncio.get_event_loop().time()

                    # Periodic heartbeat
                    if now - last_heartbeat >= heartbeat_interval:
                        duration = now - loop_start
                        oldest_age = 0.0
                        if self.item_creation_times:
                            oldest_age = now - min(self.item_creation_times.values())
                        logger.debug(
                            f"Recording heartbeat: {duration:.0f}s elapsed, "
                            f"{len(self.pending_items)} pending, oldest {oldest_age:.0f}s ago"
                        )
                        last_heartbeat = now

                    # Check for stalled transcriptions
                    stall_threshold = self.config.get("stall_alert_threshold", 10)
                    for item_id, created_at in list(self.item_creation_times.items()):
                        age = now - created_at

                        if age > stall_threshold and item_id not in alerted_items:
                            await self._alert_transcription_stall(item_id, age)
                            alerted_items.add(item_id)
            except asyncio.CancelledError:
                pass
            finally:
                logger.debug("monitor task exiting")

        # Create tasks explicitly so we can manage them
        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(recv_events())
        monitor_task = asyncio.create_task(monitor_transcriptions())

        try:
            # Wait for EITHER send or recv to complete (whichever exits first)
            # This returns immediately when any websocket task exits
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.debug(f"WebSocket task exited: done={len(done)}, pending={len(pending)}")

            # Determine exit reason if not already set by error
            if exit_reason == "unknown":
                if self.state != State.RECORDING:
                    exit_reason = "user_stopped"
                else:
                    exit_reason = "task_exited"

            # Cancel the other websocket task if still running
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            exit_reason = "cancelled"
            raise

        finally:
            # Cancel ALL tasks when exiting (handles external cancellation too)
            for task in [send_task, recv_task, monitor_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Log exit summary
            duration = asyncio.get_event_loop().time() - loop_start
            logger.info(f"Recording loop ended: {exit_reason} ({duration:.1f}s)")

        # If WebSocket error occurred during recording, handle failure
        if websocket_error and self.state == State.RECORDING:
            logger.info(f"WebSocket failed during recording, triggering failure handler")
            await self._handle_recording_failure()

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
