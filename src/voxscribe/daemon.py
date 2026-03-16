#!/usr/bin/env python3
"""
Voxscribe Daemon - Streaming speech-to-text with provider abstraction.

Runs as a systemd user service, listens on Unix socket for commands.
Commands: START, STOP, STATUS, TOGGLE

Emits DBus signals for GNOME extension integration.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from voxscribe.providers import TranscriptionProvider, create_provider
from voxscribe.silence_gate import SilenceAction, SilenceGate

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

        # Backward compatibility: migrate old format
        if "provider" not in config:
            config["provider"] = "openai"
        if "openai" not in config and "transcription" in config:
            old_t = config.pop("transcription")
            old_v = config.pop("vad", {})
            oai: dict[str, Any] = {}
            if "model" in old_t:
                oai["model"] = old_t["model"]
            if "prompt" in old_t:
                oai["prompt"] = old_t["prompt"]
            if old_t.get("language"):
                config.setdefault("language", old_t["language"])
            if "type" in old_v:
                oai["vad_type"] = old_v["type"]
            if "threshold" in old_v:
                oai["vad_threshold"] = old_v["threshold"]
            if "prefix_padding_ms" in old_v:
                oai["vad_prefix_padding_ms"] = old_v["prefix_padding_ms"]
            if "silence_duration_ms" in old_v:
                oai["vad_silence_duration_ms"] = old_v["silence_duration_ms"]
            config["openai"] = oai
            logger.info("Migrated old config format to new provider-based format")

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
        """DBus interface for status updates to GNOME extension.

        Signals carry truncated text for efficient panel display.
        GetStatus returns full text on demand (for popup/copy).
        """

        def __init__(self, text_provider: "Callable[[], str]") -> None:
            super().__init__(DBUS_NAME)
            self._state = "idle"
            self._signal_text = ""
            self._text_provider = text_provider

        @dbus_signal()
        def StateChanged(self) -> "ss":
            """Signal emitted when state changes. Returns (state, truncated_text)."""
            return [self._state, self._signal_text]

        @method()
        def GetStatus(self) -> "ss":
            """Get current state and full transcription text."""
            return [self._state, self._text_provider()]

        def emit_state(self, state: str, text: str = "") -> None:
            """Emit state change signal with truncated text for panel display."""
            self._state = state
            self._signal_text = text
            self.StateChanged()


class VoxscribeDaemon:
    """Main daemon class managing recording and transcription."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.state = State.IDLE
        self.api_key: str = ""
        self.config = config
        self.provider: Optional[TranscriptionProvider] = None
        self.silence_gate: Optional[SilenceGate] = None
        self.pw_record_proc: Optional[asyncio.subprocess.Process] = None
        self.recording_task: Optional[asyncio.Task[None]] = None
        self.shutdown_event = asyncio.Event()
        self.force_stop_event = asyncio.Event()
        self.current_output_file: Optional[Path] = None
        self.dbus_interface: Optional[Any] = None
        self.dbus_bus: Optional[Any] = None
        self._websocket_error: bool = False

    async def setup_dbus(self) -> None:
        """Set up DBus service for GNOME extension communication."""
        if not DBUS_AVAILABLE:
            logger.info("DBus not available (dbus-next not installed)")
            return

        try:
            self.dbus_bus = await MessageBus().connect()
            self.dbus_interface = VoxscribeDBusInterface(self._get_current_text)
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
        """Load API key from environment based on configured provider."""
        provider_name = self.config.get("provider", "openai")
        env_var = "OPENAI_API_KEY" if provider_name == "openai" else "ELEVENLABS_API_KEY"

        self.api_key = os.environ.get(env_var, "")
        if self.api_key:
            logger.info(f"API key loaded from {env_var}")
            return True
        logger.error(f"{env_var} environment variable not set")
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
        import subprocess

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
        import subprocess

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

    def _on_text_update(self, text: str) -> None:
        """Provider callback: transcript text updated."""
        self._write_result_file()
        self.emit_state("recording", text[-500:])

    def _on_provider_error(self, msg: str) -> None:
        """Provider callback: error occurred."""
        logger.error(f"Provider error: {msg}")
        self._websocket_error = True

    async def start_recording(self) -> tuple[bool, str]:
        """Start recording and transcription session."""
        if self.state != State.IDLE:
            return False, f"Cannot start: state is {self.state.value}"

        # Set state immediately to prevent race conditions from rapid toggles
        self.state = State.RECORDING
        self.play_sound(SOUND_START)
        self.emit_state("recording", "")
        logger.info("Starting recording session")
        self._websocket_error = False

        # Create timestamped output file and update symlink
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_output_file = OUTPUT_DIR / f"voxscribe-{timestamp}.txt"
        self.current_output_file.touch()
        RESULT_SYMLINK.unlink(missing_ok=True)
        RESULT_SYMLINK.symlink_to(self.current_output_file)
        logger.info(f"Output file: {self.current_output_file.name}")

        # Create provider
        try:
            self.provider = create_provider(self.config, self.api_key)
            self.provider.on_ready = lambda: logger.info("Provider ready")
            self.provider.on_text_update = self._on_text_update
            self.provider.on_error = self._on_provider_error
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            self.play_sound(SOUND_ERROR)
            self.emit_state("error", "")
            self.state = State.IDLE
            asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))
            return False, f"Failed to create provider: {e}"

        # Create silence gate (only active for elevenlabs when enabled)
        provider_name = self.config.get("provider", "openai")
        if provider_name == "elevenlabs":
            self.silence_gate = SilenceGate(self.config)
        else:
            self.silence_gate = None

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

        # Connect provider (handles WebSocket internally)
        try:
            await self.provider.connect()
        except Exception as e:
            logger.error(f"Failed to connect provider: {e}")
            await self._terminate_pw_record()
            self.play_sound(SOUND_ERROR)
            self.emit_state("error", "")
            self.state = State.IDLE
            asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))
            return False, f"Failed to connect to transcription service: {e}"

        # Start recording loop task
        self.recording_task = asyncio.create_task(self._recording_loop())
        logger.info("Recording started")
        return True, "Recording started"

    async def stop_recording(self) -> tuple[bool, str]:
        """Stop recording and wait for final transcription."""
        if self.state == State.IDLE:
            return False, "Not recording"
        if self.state == State.TRANSCRIBING:
            logger.info("Force stop requested - aborting transcription wait")
            self.force_stop_event.set()
            return True, "Force stopping..."

        # Clear force stop for normal stop flow
        self.force_stop_event.clear()
        logger.info("Stopping recording")
        self.play_sound(SOUND_STOP)
        self.state = State.TRANSCRIBING
        self.emit_state("transcribing", self._get_current_text()[-500:])

        # Cancel recording task first
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
        if self.provider:
            await self.provider.commit()

        # Wait for pending transcriptions
        wait_start = time.monotonic()
        safety_timeout = self.config.get("transcription_timeout", 120)
        wait_exit_reason = "completed"

        while self.provider and self.provider.has_pending() and (time.monotonic() - wait_start) < safety_timeout:
            if self.force_stop_event.is_set():
                wait_exit_reason = "force_stopped"
                break
            await asyncio.sleep(0.2)

        # Check if we exited due to timeout
        if self.provider and self.provider.has_pending() and wait_exit_reason == "completed":
            wait_exit_reason = "timeout"

        if wait_exit_reason == "completed":
            logger.info("All transcriptions complete")
        elif wait_exit_reason == "force_stopped":
            logger.info("Wait aborted by user")
        elif wait_exit_reason == "timeout":
            logger.warning(f"Timeout: exiting with pending transcriptions")
        else:
            logger.warning(f"Wait ended ({wait_exit_reason})")

        # Close provider
        if self.provider:
            await self.provider.close()

        # Process result
        result = self._get_current_text()
        logger.info(f"Final transcription: {len(result)} chars")

        if result:
            if self.current_output_file:
                self.current_output_file.write_text(result)
                logger.info(f"Saved: {self.current_output_file.name}")

            clipboard_text = f"stt-rec: {result}"
            self.copy_to_clipboard(clipboard_text)

        # Determine outcome
        has_failures = self.provider and self.provider.has_pending()
        if has_failures:
            logger.warning("Partial transcription: pending items remain")
            self.play_sound(SOUND_ERROR)
            self.emit_state("partial")
        else:
            self.play_sound(SOUND_DONE)
            self.emit_state("done")

        self.state = State.IDLE
        asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle", ""))

        status = "partial" if has_failures else "complete"
        return True, f"Transcription {status}: {len(result)} chars"

    def _get_current_text(self) -> str:
        """Get current transcription text."""
        if self.provider:
            return self.provider.get_text()
        return ""

    def _write_result_file(self) -> None:
        """Write current transcripts to output file (live updates)."""
        if not self.current_output_file:
            return
        result = self._get_current_text()
        if result:
            self.current_output_file.write_text(result)

    async def _handle_recording_failure(self) -> None:
        """Handle WebSocket failure during recording - save partial data and notify user."""
        logger.warning("Handling recording failure - saving partial data")

        # Stop pw-record
        await self._terminate_pw_record()

        # Close provider
        if self.provider:
            await self.provider.close()

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
        self.emit_state("partial")
        self.state = State.IDLE
        asyncio.get_event_loop().call_later(5, lambda: self.emit_state("idle"))

    async def _recording_loop(self) -> None:
        """Main loop for sending audio and monitoring health.

        The provider's recv loop runs internally as its own task.
        """
        logger.info("Recording loop started")
        loop_start = time.monotonic()
        exit_reason = "unknown"

        async def send_audio() -> None:
            """Read pw-record and send chunks through silence gate to provider."""
            nonlocal exit_reason
            logger.debug("send_audio task started")
            try:
                while self.state == State.RECORDING and self.pw_record_proc and self.provider:
                    try:
                        assert self.pw_record_proc.stdout is not None
                        chunk = await asyncio.wait_for(
                            self.pw_record_proc.stdout.read(CHUNK_BYTES),
                            timeout=0.2,
                        )
                        if not chunk:
                            break

                        if self.silence_gate:
                            action, onset_chunks = self.silence_gate.process(chunk)
                            if action == SilenceAction.SEND:
                                for onset_chunk in onset_chunks:
                                    await self.provider.send_audio(onset_chunk)
                                await self.provider.send_audio(chunk)
                            elif action == SilenceAction.KEEPALIVE:
                                await self.provider.send_audio(chunk)
                            # SKIP: do nothing
                        else:
                            await self.provider.send_audio(chunk)

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Send audio error: {e}")
                        self._websocket_error = True
                        exit_reason = "send_audio_error"
                        break
            finally:
                logger.debug("send_audio task exiting")

        async def monitor() -> None:
            """Watchdog and heartbeat monitoring."""
            logger.debug("monitor task started")
            heartbeat_interval = 30
            last_heartbeat = time.monotonic()
            watchdog_timeout = 30.0
            try:
                while self.state == State.RECORDING:
                    await asyncio.sleep(5)
                    now = time.monotonic()

                    # Periodic heartbeat
                    if now - last_heartbeat >= heartbeat_interval:
                        duration = now - loop_start
                        logger.debug(
                            f"Recording heartbeat: {duration:.0f}s elapsed"
                        )
                        last_heartbeat = now

                    # Watchdog: check provider responsiveness (skip during silence gaps)
                    if self.provider:
                        in_silence_gap = self.silence_gate and self.silence_gate.in_gap
                        if not in_silence_gap:
                            silence = now - self.provider.last_event_time
                            if silence > watchdog_timeout:
                                logger.warning(
                                    f"No provider events for {silence:.0f}s, possible stall"
                                )
                                self.play_sound(SOUND_ERROR)
                                self.emit_state("partial", "Connection may be stalled")

                    # Check for provider error
                    if self._websocket_error:
                        break

            except asyncio.CancelledError:
                pass
            finally:
                logger.debug("monitor task exiting")

        send_task = asyncio.create_task(send_audio())
        monitor_task = asyncio.create_task(monitor())

        try:
            # Wait for send_audio to exit (recv runs inside provider)
            await send_task

            if exit_reason == "unknown":
                if self.state != State.RECORDING:
                    exit_reason = "user_stopped"
                else:
                    exit_reason = "task_exited"

        except asyncio.CancelledError:
            exit_reason = "cancelled"
            raise

        finally:
            for task in [send_task, monitor_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            duration = time.monotonic() - loop_start
            logger.info(f"Recording loop ended: {exit_reason} ({duration:.1f}s)")

        # If error occurred during recording, handle failure
        if self._websocket_error and self.state == State.RECORDING:
            logger.info("Provider failed during recording, triggering failure handler")
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

        if self.provider:
            await self.provider.close()
            self.provider = None

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
