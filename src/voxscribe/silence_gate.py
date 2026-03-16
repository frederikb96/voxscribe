"""
Client-side silence detection for Voxscribe.

Filters audio chunks to avoid sending pure silence, reducing bandwidth and
improving transcription quality with providers that benefit from it.
"""

import logging
import struct
import time
from enum import Enum
from typing import Any

logger = logging.getLogger("voxscribe")

SAMPLE_RATE = 24000
CHUNK_BYTES = 4800  # 100ms at 24kHz 16-bit mono
KEEPALIVE_INTERVAL = 15.0
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
    """Client-side silence gate with EMA smoothing and onset buffering."""

    def __init__(self, config: dict[str, Any]) -> None:
        gate_config = config.get("silence_gate", {})
        self.enabled = gate_config.get("enabled", False)
        self.threshold = gate_config.get("threshold", 0.010)
        self.gap_seconds = gate_config.get("gap_seconds", 3.0)

        self._ema: float = 0.0
        self._in_gap: bool = True
        self._gap_start: float = time.monotonic()
        self._last_keepalive: float = time.monotonic()
        self._speech_count: int = 0
        self._onset_buffer: list[bytes] = []

    @property
    def in_gap(self) -> bool:
        return self._in_gap

    def process(self, chunk: bytes) -> tuple[SilenceAction, list[bytes]]:
        """Process a chunk, return action and any onset chunks to flush.

        Returns:
            (action, onset_chunks): action is SEND/SKIP/KEEPALIVE,
            onset_chunks is non-empty only on speech onset confirmation.
        """
        if not self.enabled:
            return SilenceAction.SEND, []

        rms = _rms(chunk)
        self._ema = EMA_ALPHA * rms + (1 - EMA_ALPHA) * self._ema
        now = time.monotonic()
        is_loud = self._ema > self.threshold

        if is_loud:
            if self._in_gap:
                # Potential speech onset
                self._speech_count += 1
                self._onset_buffer.append(chunk)

                if self._speech_count >= SPEECH_CONFIRM_CHUNKS:
                    # Confirmed speech: flush onset buffer
                    self._in_gap = False
                    onset_chunks = list(self._onset_buffer)
                    self._onset_buffer.clear()
                    self._speech_count = 0
                    logger.debug("Silence gate: speech onset confirmed")
                    return SilenceAction.SEND, onset_chunks
                else:
                    return SilenceAction.SKIP, []
            else:
                # Already in speech
                return SilenceAction.SEND, []
        else:
            # Below threshold
            self._speech_count = 0
            self._onset_buffer.clear()

            if not self._in_gap:
                # Transition to gap
                self._in_gap = True
                self._gap_start = now
                self._last_keepalive = now
                logger.debug("Silence gate: entering gap")

            # Check keepalive
            if now - self._last_keepalive >= KEEPALIVE_INTERVAL:
                self._last_keepalive = now
                return SilenceAction.KEEPALIVE, []

            return SilenceAction.SKIP, []
