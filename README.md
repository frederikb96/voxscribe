# Voxscribe

Real-time speech-to-text using OpenAI Realtime or ElevenLabs Scribe v2. Runs as a systemd user daemon with keyboard shortcut control.

**Features:**
- Stream transcription to file as you speak
- Automatic clipboard copy on completion, verified by reading the clipboard back
- GNOME panel indicator with full-text popup
- Voice activity detection for natural pauses

## Installation

Requires Python 3.11+, PipeWire, and wl-clipboard (Wayland).

```bash
# Install with pipx (recommended)
pipx install git+https://github.com/frederikb/voxscribe.git

# Set up systemd service
voxscribe setup
```

**API key:** Set `OPENAI_API_KEY` environment variable (e.g., in `~/.bashrc`)

## Usage

```bash
voxscribe toggle    # Start/stop recording (bind this to a keyboard shortcut)
voxscribe status    # Check if daemon is running
voxscribe start     # Start recording
voxscribe stop      # Stop recording
```

**Output:**
- Clipboard: `stt-rec: <transcription>`
- File: `~/.tmp/voxscribe-YYYYMMDD-HHMMSS.txt`
- Live preview: `tail -f ~/.tmp/voxscribe-result.txt`

## Clipboard

The daemon hands the text to the GNOME Shell extension over DBus, so the compositor itself owns
the clipboard: no helper process that can die, no X11 chunked transfer, and the text survives a
daemon restart. Without the extension it falls back to `wl-copy`. Every delivery is read back with
`wl-paste` and compared; the journal line `Clipboard: N chars via shell|wl-copy` reports which
path delivered it, and the indicator shows `Error!` instead of `Copied!` when none did.

## GNOME Extension

Shows recording status in the top panel, a scrollable popup with the full text, a copy button and
the clipboard service described above.

```bash
voxscribe install-extension     # From the repo folder; log out/in on Wayland to load new code
gnome-extensions enable voxscribe@frederikb.github.com
```

## Configuration

Edit `~/.config/voxscribe/config.yaml`; all options with defaults are in `config.example.yaml`.

## Uninstall

```bash
voxscribe teardown
pipx uninstall voxscribe
```

## Requirements

- **PipeWire:** `pw-record` for audio capture, `pw-play` for sound feedback
- **wl-clipboard:** `wl-paste` for clipboard verification, `wl-copy` as fallback writer
- **API key:** `OPENAI_API_KEY` or `ELEVENLABS_API_KEY`, matching the configured provider

## Tests

```bash
python -m unittest discover -s tests
```
