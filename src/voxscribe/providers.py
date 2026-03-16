"""
Transcription providers for Voxscribe.

Abstract base class + OpenAI Realtime and ElevenLabs Scribe v2 implementations.
"""

import asyncio
import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import websockets

logger = logging.getLogger("voxscribe")

SAMPLE_RATE = 24000


class TranscriptionProvider(ABC):
    """Base class for streaming transcription providers."""

    def __init__(self) -> None:
        self.on_ready: Optional[Callable[[], None]] = None
        self.on_text_update: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.last_event_time: float = time.monotonic()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task[None]] = None

    @abstractmethod
    async def connect(self) -> None:
        """Open WebSocket, configure session, start internal recv task."""

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Format and send one PCM chunk."""

    @abstractmethod
    async def commit(self) -> None:
        """Force-flush the audio buffer."""

    @abstractmethod
    def get_text(self) -> str:
        """Return current full transcript."""

    @abstractmethod
    def has_pending(self) -> bool:
        """Whether there are unresolved transcriptions."""

    @abstractmethod
    def reset(self) -> None:
        """Clear transcript state."""

    async def close(self) -> None:
        """Cancel recv task, close WebSocket."""
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            try:
                await asyncio.wait_for(self._ws.close(), timeout=1)
            except Exception:
                pass
            self._ws = None
            logger.info("WebSocket closed")


class OpenAiProvider(TranscriptionProvider):
    """OpenAI Realtime Transcription API provider."""

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        super().__init__()
        self._api_key = api_key
        self._config = config
        self.transcripts: dict[str, str] = {}
        self.pending_items: set[str] = set()
        self.failed_items: set[str] = set()
        self.item_creation_times: dict[str, float] = {}

    async def connect(self) -> None:
        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers, max_size=None),
            timeout=10,
        )
        logger.info("WebSocket connected")

        # Wait for session.created
        msg = await asyncio.wait_for(self._ws.recv(), timeout=5)
        ev = json.loads(msg)
        logger.debug(f"Received: {ev.get('type')}")
        self.last_event_time = time.monotonic()

        # Build session config
        oai = self._config.get("openai", {})

        if "model" not in oai:
            raise ValueError("openai.model is required in config")

        transcription_settings: dict[str, Any] = {"model": oai["model"]}
        if oai.get("prompt"):
            transcription_settings["prompt"] = oai["prompt"]

        language = self._config.get("language")
        if language:
            transcription_settings["language"] = language

        turn_detection: dict[str, Any] = {"type": oai.get("vad_type", "server_vad")}
        if "vad_threshold" in oai:
            turn_detection["threshold"] = oai["vad_threshold"]
        if "vad_prefix_padding_ms" in oai:
            turn_detection["prefix_padding_ms"] = oai["vad_prefix_padding_ms"]
        if "vad_silence_duration_ms" in oai:
            turn_detection["silence_duration_ms"] = oai["vad_silence_duration_ms"]

        session_config: dict[str, Any] = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription_settings,
            "turn_detection": turn_detection,
        }

        logger.info(f"VAD: {turn_detection.get('type')}, config keys: {list(turn_detection.keys())}")
        if oai.get("prompt"):
            logger.debug(f"Prompt: {oai['prompt'][:80]}...")

        await self._ws.send(
            json.dumps({"type": "transcription_session.update", "session": session_config})
        )
        msg = await asyncio.wait_for(self._ws.recv(), timeout=5)
        ev = json.loads(msg)
        logger.debug(f"Session configured: {ev.get('type')}")
        self.last_event_time = time.monotonic()

        # Start recv task
        self._recv_task = asyncio.create_task(self._recv_loop())

        if self.on_ready:
            self.on_ready()

    async def _recv_loop(self) -> None:
        """Receive and handle WebSocket events."""
        logger.debug("OpenAI recv task started")
        try:
            while self._ws:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=0.2)
                    self.last_event_time = time.monotonic()
                    self._handle_event(json.loads(msg))
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except websockets.ConnectionClosed:
                    logger.warning("WebSocket closed during recv")
                    if self.on_error:
                        self.on_error("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Recv event error: {e}")
                    if self.on_error:
                        self.on_error(f"Recv error: {e}")
                    break
        finally:
            logger.debug("OpenAI recv task exiting")

    def _handle_event(self, ev: dict[str, Any]) -> None:
        t = ev.get("type", "")
        item_id = ev.get("item_id", "")

        if t == "input_audio_buffer.speech_started":
            if item_id:
                self.pending_items.add(item_id)

        elif t == "conversation.item.input_audio_transcription.delta":
            delta = ev.get("delta", "")
            if item_id and delta:
                self.transcripts[item_id] = self.transcripts.get(item_id, "") + delta
                if self.on_text_update:
                    self.on_text_update(self.get_text())

        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = ev.get("transcript", "")
            if item_id:
                if transcript:
                    self.transcripts[item_id] = transcript
                self.pending_items.discard(item_id)
                created_at = self.item_creation_times.pop(item_id, None)
                if created_at:
                    latency = time.monotonic() - created_at
                    logger.info(
                        f"Transcription completed [{item_id[:8]}]: "
                        f"{len(transcript)} chars ({latency:.1f}s latency)"
                    )
                else:
                    logger.info(f"Transcription completed [{item_id[:8]}]: {len(transcript)} chars")
                if self.on_text_update:
                    self.on_text_update(self.get_text())

        elif t == "input_audio_buffer.committed":
            committed_item_id = ev.get("item_id", "")
            if committed_item_id:
                self.pending_items.add(committed_item_id)
                self.item_creation_times[committed_item_id] = time.monotonic()
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
            error_msg = str(ev.get("error", {}))
            logger.error(f"API error: {error_msg}")
            if self.on_error:
                self.on_error(f"API error: {error_msg}")

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )

    async def commit(self) -> None:
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                logger.info("Sent audio commit")
            except Exception as e:
                logger.error(f"Failed to send commit: {e}")

    def get_text(self) -> str:
        return " ".join(self.transcripts[k] for k in sorted(self.transcripts)).strip()

    def has_pending(self) -> bool:
        return bool(self.pending_items)

    def reset(self) -> None:
        self.transcripts.clear()
        self.pending_items.clear()
        self.failed_items.clear()
        self.item_creation_times.clear()


class ElevenLabsProvider(TranscriptionProvider):
    """ElevenLabs Scribe v2 Realtime provider."""

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        super().__init__()
        self._api_key = api_key
        self._config = config
        self.committed_segments: list[str] = []
        self.current_partial: str = ""
        self.has_uncommitted_partial: bool = False

    async def connect(self) -> None:
        el = self._config.get("elevenlabs", {})

        params = [
            f"model_id=scribe_v2_realtime",
            f"audio_format=pcm_{SAMPLE_RATE}",
            f"commit_strategy=vad",
        ]
        if "vad_silence_threshold_secs" in el:
            params.append(f"vad_silence_threshold_secs={el['vad_silence_threshold_secs']}")
        if "vad_threshold" in el:
            params.append(f"vad_threshold={el['vad_threshold']}")
        if "enable_logging" in el:
            params.append(f"enable_logging={'true' if el['enable_logging'] else 'false'}")

        language = self._config.get("language")
        if language:
            params.append(f"language_code={language}")

        url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{'&'.join(params)}"
        headers = {"xi-api-key": self._api_key}

        self._ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers, max_size=None),
            timeout=10,
        )
        logger.info("WebSocket connected")

        # Wait for session_started
        msg = await asyncio.wait_for(self._ws.recv(), timeout=5)
        ev = json.loads(msg)
        if ev.get("message_type") == "session_started":
            logger.info(f"ElevenLabs session started: {ev.get('session_id', 'unknown')}")
        else:
            logger.warning(f"Expected session_started, got: {ev.get('message_type')}")
        self.last_event_time = time.monotonic()

        # Start recv task
        self._recv_task = asyncio.create_task(self._recv_loop())

        if self.on_ready:
            self.on_ready()

    async def _recv_loop(self) -> None:
        """Receive and handle WebSocket events."""
        logger.debug("ElevenLabs recv task started")
        try:
            while self._ws:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=0.2)
                    self.last_event_time = time.monotonic()
                    ev = json.loads(msg)
                    self._handle_event(ev)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except websockets.ConnectionClosed:
                    logger.warning("WebSocket closed during recv")
                    if self.on_error:
                        self.on_error("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Recv event error: {e}")
                    if self.on_error:
                        self.on_error(f"Recv error: {e}")
                    break
        finally:
            logger.debug("ElevenLabs recv task exiting")

    def _handle_event(self, ev: dict[str, Any]) -> None:
        mt = ev.get("message_type", "")

        if mt == "partial_transcript":
            self.current_partial = ev.get("text", "")
            self.has_uncommitted_partial = bool(self.current_partial)
            if self.on_text_update:
                self.on_text_update(self.get_text())

        elif mt in ("committed_transcript", "committed_transcript_with_timestamps"):
            text = ev.get("text", "")
            if text:
                self.committed_segments.append(text)
                logger.info(f"Committed transcript: {len(text)} chars")
            self.current_partial = ""
            self.has_uncommitted_partial = False
            if self.on_text_update:
                self.on_text_update(self.get_text())

        elif mt == "commit_throttled":
            logger.warning("Commit throttled by ElevenLabs")

        elif mt == "insufficient_audio_activity":
            logger.debug("Insufficient audio activity")

        elif mt == "session_started":
            pass

        else:
            # Treat any other message_type as a potential error
            if "error" in mt.lower():
                error_msg = ev.get("message", ev.get("error", str(ev)))
                logger.error(f"ElevenLabs error ({mt}): {error_msg}")
                if self.on_error:
                    self.on_error(f"ElevenLabs error ({mt}): {error_msg}")
            else:
                logger.debug(f"Unhandled ElevenLabs event: {mt}")

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws:
            await self._ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(chunk).decode(),
                        "commit": False,
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            )

    async def commit(self) -> None:
        """Send a short silent chunk with commit=true to force flush."""
        if self._ws:
            try:
                # 10ms of silence at 24kHz 16-bit mono = 480 bytes
                silent_chunk = b"\x00" * 480
                await self._ws.send(
                    json.dumps(
                        {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": base64.b64encode(silent_chunk).decode(),
                            "commit": True,
                            "sample_rate": SAMPLE_RATE,
                        }
                    )
                )
                logger.info("Sent commit with silent chunk")
            except Exception as e:
                logger.error(f"Failed to send commit: {e}")

    def get_text(self) -> str:
        parts = list(self.committed_segments)
        if self.current_partial:
            parts.append(self.current_partial)
        return " ".join(parts).strip()

    def has_pending(self) -> bool:
        return self.has_uncommitted_partial

    def reset(self) -> None:
        self.committed_segments.clear()
        self.current_partial = ""
        self.has_uncommitted_partial = False


def create_provider(config: dict[str, Any], api_key: str) -> TranscriptionProvider:
    """Factory: create the right provider based on config."""
    provider_name = config.get("provider", "openai")

    if provider_name == "openai":
        return OpenAiProvider(api_key, config)
    elif provider_name == "elevenlabs":
        return ElevenLabsProvider(api_key, config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
