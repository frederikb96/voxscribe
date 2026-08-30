"""Filesystem locations shared by the daemon and the CLI."""

from pathlib import Path

# Transcripts and recordings carry whatever was spoken, so they live in a
# private directory rather than world-readable /tmp.
OUTPUT_DIR = Path.home() / ".tmp"
RESULT_SYMLINK = OUTPUT_DIR / "voxscribe-result.txt"
RECORDINGS_DIR = OUTPUT_DIR / "voxscribe-recordings"


def ensure_output_dir() -> None:
    """Create OUTPUT_DIR owner-only. An existing directory keeps its mode."""
    OUTPUT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
