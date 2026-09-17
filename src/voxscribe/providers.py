"""
Transcription provider for Voxscribe.

Streams audio to ElevenLabs Scribe v2 Realtime and assembles the transcript.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Callable, Optional

import websockets

logger = logging.getLogger("voxscribe")

SAMPLE_RATE = 24000

# Shortest repeated stretch treated as a duplicate; shorter repeats are normal speech.
MIN_OVERLAP = 30


def strip_overlap(existing: str, incoming: str) -> str:
    """Drop the leading part of incoming that repeats the tail of existing.

    A forced commit can re-emit audio that a VAD commit already transcribed, which lands as a
    segment repeating the end of the text so far. Returns "" when incoming is entirely a repeat.
    """
    tail = existing.rstrip()
    head = incoming.strip()
    for n in range(min(len(tail), len(head)), MIN_OVERLAP - 1, -1):
        if tail.endswith(head[:n]):
            return head[n:].lstrip()
    return incoming


class ElevenLabsProvider:
    """Streams audio to ElevenLabs Scribe v2 Realtime and assembles the transcript."""

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        self.on_ready: Optional[Callable[[], None]] = None
        self.on_text_update: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.last_event_time: float = time.monotonic()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task[None]] = None
        self._api_key = api_key
        self._config = config
        self.committed_segments: list[str] = []
        self.current_partial: str = ""
        self._pending_commit: bool = False
        self.close_code: Optional[int] = None
        self.close_reason: str = ""
        self._previous_text: str = ""
        self._first_chunk_sent: bool = False

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

        language = self._config.get("language", "")
        if language and language != "auto":
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
        self._first_chunk_sent = False

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
                except websockets.ConnectionClosed as e:
                    self.close_code = e.code
                    self.close_reason = e.reason or ""
                    self._pending_commit = False
                    logger.warning(f"WebSocket closed during recv: {e.code} {self.close_reason}")
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
            if self.on_text_update:
                self.on_text_update(self.get_text())

        elif mt in ("committed_transcript", "committed_transcript_with_timestamps"):
            text = ev.get("text", "")
            if text:
                kept = strip_overlap(" ".join(self.committed_segments), text)
                if len(kept) < len(text):
                    logger.info(f"Dropped {len(text) - len(kept)} duplicated chars from commit")
                if kept:
                    self.committed_segments.append(kept)
                    logger.info(f"Committed transcript: {len(kept)} chars")
            self.current_partial = ""
            self._pending_commit = False
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

    def set_previous_text(self, text: str) -> None:
        """Set context text for the first audio chunk (max 50 chars per ElevenLabs docs)."""
        self._previous_text = text[-50:] if text else ""

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws:
            msg: dict[str, Any] = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode(),
                "commit": False,
                "sample_rate": SAMPLE_RATE,
            }
            if not self._first_chunk_sent and self._previous_text:
                msg["previous_text"] = self._previous_text
            self._first_chunk_sent = True
            await self._ws.send(json.dumps(msg))

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
                self._pending_commit = True
                logger.info("Sent commit with silent chunk")
            except Exception as e:
                logger.error(f"Failed to send commit: {e}")

    def get_text(self) -> str:
        parts = list(self.committed_segments)
        if self.current_partial:
            parts.append(self.current_partial)
        return " ".join(parts).strip()

    def has_pending(self) -> bool:
        return self._pending_commit

    def reset(self) -> None:
        self.committed_segments.clear()
        self.current_partial = ""
        self._pending_commit = False
        self.close_code = None
        self.close_reason = ""
        self._first_chunk_sent = False

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
