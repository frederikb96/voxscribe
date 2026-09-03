"""Clipboard delivery shared by the daemon and the CLI.

Order of delivery: the GNOME Shell extension (the compositor holds the text in memory, so it
survives daemon restarts and involves no X11 chunked transfer), then wl-copy. Every delivery is
verified by reading the clipboard back with wl-paste.
"""

import asyncio
import logging
from typing import Optional

from dbus_next import Message, MessageType
from dbus_next.aio import MessageBus

logger = logging.getLogger("voxscribe")

CLIPBOARD_PREFIX = "stt-rec: "

SHELL_NAME = "com.github.frederikb.Voxscribe.Shell"
SHELL_PATH = "/com/github/frederikb/Voxscribe/Shell"
SHELL_INTERFACE = SHELL_NAME

DBUS_TIMEOUT = 3
PROCESS_TIMEOUT = 5


def clipboard_payload(text: str) -> str:
    """The exact string that lands on the clipboard for a transcription."""
    return f"{CLIPBOARD_PREFIX}{text}"


async def copy_text(text: str, bus: Optional[MessageBus] = None) -> str:
    """Put text on the clipboard and return the method that delivered it.

    Tries the GNOME Shell extension first, then wl-copy. Each attempt is verified by reading the
    clipboard back; an unverifiable read-back (wl-paste missing) is accepted with a warning.
    Raises RuntimeError when no method delivered the text.
    """
    for method in ("shell", "wl-copy"):
        try:
            if method == "shell":
                await _copy_via_shell(text, bus)
            else:
                await _copy_via_wl_copy(text)
        except Exception as e:
            logger.warning(f"Clipboard via {method} failed: {e}")
            continue

        verified = await _read_back_matches(text)
        if verified is False:
            logger.warning(f"Clipboard via {method}: read-back does not match")
            continue
        suffix = "" if verified else " (read-back unavailable)"
        logger.info(f"Clipboard: {len(text)} chars via {method}{suffix}")
        return method

    raise RuntimeError("no clipboard method delivered the text")


async def _copy_via_shell(text: str, bus: Optional[MessageBus]) -> None:
    """Ask the GNOME Shell extension to take clipboard ownership of text."""
    own_bus = bus is None
    if bus is None:
        bus = await MessageBus().connect()
    try:
        call = bus.call(
            Message(
                destination=SHELL_NAME,
                path=SHELL_PATH,
                interface=SHELL_INTERFACE,
                member="Copy",
                signature="s",
                body=[text],
            )
        )
        reply = await asyncio.wait_for(call, timeout=DBUS_TIMEOUT)
        if reply is None or reply.message_type == MessageType.ERROR:
            detail = reply.body[0] if reply and reply.body else "no reply"
            raise RuntimeError(f"{reply.error_name if reply else 'error'}: {detail}")
        if not reply.body or reply.body[0] is not True:
            raise RuntimeError("extension reported failure")
    finally:
        if own_bus:
            bus.disconnect()


async def _copy_via_wl_copy(text: str) -> None:
    """Hand text to wl-copy, which forks a background process serving the selection.

    stdout/stderr must not be pipes: the forked child inherits them and would keep
    communicate() waiting until the clipboard changes again.
    """
    proc = await asyncio.create_subprocess_exec(
        "wl-copy",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        await asyncio.wait_for(proc.communicate(text.encode()), timeout=PROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("wl-copy timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"wl-copy exited with {proc.returncode}")


async def _read_back_matches(text: str) -> Optional[bool]:
    """Read the clipboard with wl-paste and compare. None when wl-paste is unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "wl-paste",
            "--no-newline",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=PROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return False
    if proc.returncode != 0:
        return False
    return out == text.encode()
