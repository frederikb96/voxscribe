# Long STT

Long-form speech-to-text via OpenAI Realtime Transcription API. Systemd daemon with Unix socket IPC.

## Quick Reference

```bash
# Control
long-stt-ctl start    # begin recording
long-stt-ctl stop     # stop and transcribe
long-stt-ctl toggle   # start if idle, stop if recording
long-stt-ctl status   # check state

# Logs
journalctl --user -u long-stt -f

# Deploy/destroy
ansible-playbook ansible/deploy.yml
ansible-playbook ansible/destroy.yml
```

## Output

- **Clipboard:** `stt-rec: <transcription>` via wl-copy
- **File:** `/tmp/long-stt-YYYYMMDD-HHMMSS.txt` (timestamped, created at recording start)
- **Symlink:** `/tmp/long-stt-result.txt` → current recording (for `tail -f`)

## Configuration

Edit `config.yaml` (deployed to `~/.local/lib/long-stt/config.yaml`):

```yaml
log_level: info  # debug, info, warning, error

vad:
  type: server_vad
  threshold: 0.5
  prefix_padding_ms: 300
  silence_duration_ms: 1000  # longer = allows pauses

transcription:
  model: gpt-4o-transcribe
  prompt: "Transcribe exactly..."  # verbatim mode
  language: ""  # auto-detect
```

## Files

- `~/.local/bin/long-stt-ctl` - client symlink
- `~/.local/lib/long-stt/` - daemon + venv + config
- `~/.config/systemd/user/long-stt.service`
- `/run/user/$UID/long-stt.sock` - IPC socket
- `~/.claude/.env` - needs OPENAI_API_KEY

## Gotchas

- **WAYLAND_DISPLAY must be imported** - deploy.yml handles this, but if wl-copy fails check `systemctl --user show-environment`
- **VAD buffers until silence** - turn_detection waits for silence_duration_ms before completing turn. Manual stop sends commit to force transcription of remaining audio
- **wl-copy daemon** - wl-copy forks to serve clipboard. Code uses `start_new_session=True` to avoid FD inheritance issues

## Dependencies

PipeWire (pw-record, pw-play), wl-copy, Python 3.11+, websockets
