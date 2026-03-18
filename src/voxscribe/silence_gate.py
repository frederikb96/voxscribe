"""
Client-side silence detection for Voxscribe.

Filters audio chunks to avoid sending pure silence, reducing costs
with providers that charge for all streamed audio (e.g., ElevenLabs).
"""

import logging
import struct
import time
from enum import Enum
from typing import Any

logger = logging.getLogger("voxscribe")

SAMPLE_RATE = 24000
CHUNK_BYTES = 4800  # 100ms at 24kHz 16-bit mono
KEEPALIVE_INTERVAL = 10.0
SPEECH_CONFIRM_CHUNKS = 3
EMA_ALPHA = 0.3


class SilenceAction(Enum):
    SEND = "send"
    SKIP = "skip"
    KEEPALIVE = "keepalive"


def _rms(chunk: bytes) -> float:
    """Calculate RMS amplitude from raw PCM16 little-endian bytes."""
    n_samples = len(chunk) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", chunk[: n_samples * 2])
    sum_sq = sum(s * s for s in samples)
    return (sum_sq / n_samples) ** 0.5 / 32768.0


class SilenceGate:
    """Client-side silence gate with EMA smoothing and onset buffering.

    Flow:
    - NORMAL: all chunks sent. RMS drops below threshold → start counting.
    - After gap_seconds of continuous silence → GAP mode (stop sending).
    - In GAP: send keepalive every 15s. When speech confirmed (3 consecutive
      above-threshold chunks) → back to NORMAL, flush onset buffer.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gate_config = config.get("silence_gate", {})
        self.enabled = gate_config.get("enabled", False)
        self.threshold = gate_config.get("threshold", 0.010)
        self.gap_seconds = gate_config.get("gap_seconds", 3.0)

        self._ema: float = 0.0
        self._in_gap: bool = False
        self._silence_start: float = 0.0
        self._last_keepalive: float = 0.0
        self._speech_count: int = 0
        self._onset_buffer: list[bytes] = []

        if self.enabled:
            logger.info(
                f"Silence gate enabled: threshold={self.threshold:.4f}, "
                f"gap={self.gap_seconds:.1f}s"
            )

    @property
    def in_gap(self) -> bool:
        return self._in_gap

    def process(self, chunk: bytes) -> tuple[SilenceAction, list[bytes]]:
        """Process a chunk, return action and any onset chunks to flush."""
        if not self.enabled:
            return SilenceAction.SEND, []

        rms = _rms(chunk)
        self._ema = EMA_ALPHA * rms + (1 - EMA_ALPHA) * self._ema
        now = time.monotonic()
        is_loud = self._ema > self.threshold

        if is_loud:
            self._silence_start = 0.0

            if self._in_gap:
                self._speech_count += 1
                self._onset_buffer.append(chunk)

                if self._speech_count >= SPEECH_CONFIRM_CHUNKS:
                    self._in_gap = False
                    onset_chunks = list(self._onset_buffer)
                    self._onset_buffer.clear()
                    self._speech_count = 0
                    logger.info(
                        f"Silence gate OPEN — speech resumed "
                        f"(rms={self._ema:.4f}, flushing {len(onset_chunks)} chunks)"
                    )
                    return SilenceAction.SEND, onset_chunks
                else:
                    return SilenceAction.SKIP, []
            else:
                return SilenceAction.SEND, []
        else:
            self._speech_count = 0
            self._onset_buffer.clear()

            if self._in_gap:
                if now - self._last_keepalive >= KEEPALIVE_INTERVAL:
                    self._last_keepalive = now
                    return SilenceAction.KEEPALIVE, []
                return SilenceAction.SKIP, []
            else:
                if self._silence_start == 0.0:
                    self._silence_start = now
                elapsed = now - self._silence_start
                if elapsed >= self.gap_seconds:
                    self._in_gap = True
                    self._last_keepalive = now
                    logger.info(
                        f"Silence gate CLOSED — silence for {elapsed:.1f}s "
                        f"(rms={self._ema:.4f}, threshold={self.threshold:.4f})"
                    )
                    return SilenceAction.KEEPALIVE, []
                return SilenceAction.SEND, []
