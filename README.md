# Voxscribe

Real-time speech-to-text using OpenAI's Realtime Transcription API. Runs as a systemd user daemon with keyboard shortcut control.

**Features:**
- Stream transcription to file as you speak
- Automatic clipboard copy on completion
- GNOME panel indicator (optional)
- Voice activity detection for natural pauses

## Installation

Requires Python 3.11+, PipeWire, and wl-copy (Wayland).

```bash
# Install with pipx (recommended)
pipx install git+https://github.com/frederikb/voxscribe.git

# Set up systemd service
voxscribe setup
```

**API key:** Add `OPENAI_API_KEY=sk-...` to `~/.claude/.env`

## Usage

```bash
voxscribe toggle    # Start/stop recording (bind this to a keyboard shortcut)
voxscribe status    # Check if daemon is running
voxscribe start     # Start recording
voxscribe stop      # Stop recording
```

**Output:**
- Clipboard: `stt-rec: <transcription>`
- File: `/tmp/voxscribe-YYYYMMDD-HHMMSS.txt`
- Live preview: `tail -f /tmp/voxscribe-result.txt`

## GNOME Extension (Optional)

Shows recording status in the top panel.

```bash
# Install extension
cp -r extension ~/.local/share/gnome-shell/extensions/voxscribe@frederikb.github.com

# Enable (requires GNOME Shell restart on Wayland - log out/in)
gnome-extensions enable voxscribe@frederikb.github.com
```

## Configuration

Edit `~/.config/voxscribe/config.yaml`:

```yaml
vad:
  silence_duration_ms: 1500  # Pause length before turn completion

transcription:
  model: gpt-4o-transcribe   # Or whisper-1 for literal transcription
  prompt: "..."              # Guide transcription behavior
  language: ""               # Auto-detect, or "en"/"de" hint
```

## Uninstall

```bash
voxscribe teardown
pipx uninstall voxscribe
```

## Requirements

- **PipeWire:** `pw-record` for audio capture, `pw-play` for sound feedback
- **wl-copy:** Wayland clipboard
- **OpenAI API key:** With access to Realtime API
