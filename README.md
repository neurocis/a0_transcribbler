# a0_transcribbler

**📣 Voice messages become text conversations!** This plugin enables users to send
voice memos via Telegram, Signal, WebUI, or any input channel and have them
automatically transcribed and processed by the LLM as if they were typed text.

No more typing long messages — just speak and the AI understands.

---

Automatic audio transcription plugin for Agent Zero. Transcribes audio file
attachments, audio URLs, and YouTube video audio **before** LLM processing,
injecting the transcription text directly into the conversation context.

## Features

- **Voice Memo Transcription** — Send voice messages via Telegram, Signal, or
  any platform and they're automatically converted to text for the LLM. Perfect
  for hands-free interaction and accessibility.

- **Audio File Transcription** — Automatically detects and transcribes audio
  attachments (voice messages, audio files) sent via Telegram, WebUI, or any
  other input channel. Supports `.ogg`, `.mp3`, `.wav`, `.m4a`, `.opus`,
  `.flac`, `.aac`, `.wma`, `.webm`, and more.

- **Audio URL Transcription** — Scans URLs in messages and probes their MIME
  type via HTTP HEAD request. URLs serving `audio/*` content are automatically
  downloaded and transcribed. YouTube URLs are handled separately with dedicated
  subtitle-first logic.

- **YouTube Video Transcription** — Detects YouTube URLs in messages, first
  attempts to fetch existing subtitles/captions, then falls back to downloading
  the audio track via `yt-dlp` and transcribing with Whisper. Includes
  configurable duration limits to prevent excessively long downloads.

- **Pre-LLM Injection** — Transcriptions are prepended to the user message
  before the LLM sees it, so the agent can reason about audio content naturally.

- **Reuses Existing STT** — Built on Agent Zero's built-in OpenAI Whisper
  engine. No additional STT models or API keys required.

## How It Works

1. When a user sends a message, the plugin intercepts it via the
   `hist_add_user_message` extensible hook (before history is written).
2. It scans attachments for audio file extensions.
3. It scans the message text for non-YouTube URLs and probes each with an HTTP
   HEAD request to detect `audio/*` MIME types.
4. It scans the message text for YouTube URLs.
5. Detected audio is converted to WAV (16kHz mono) via `ffmpeg` and
   transcribed using the Whisper model configured in Agent Zero settings.
6. Transcription text is prepended to the original message with a clear label.
7. The LLM receives the enriched message and can respond to the audio content.

## Requirements

- **ffmpeg** — Pre-installed in Agent Zero Docker images.
- **OpenAI Whisper** — Pre-installed in Agent Zero (`openai-whisper` package).
- **yt-dlp** — Installed automatically by the plugin's `hooks.py` on first
  install. Required only for YouTube transcription.

## Configuration

All settings are accessible from the Agent Zero Plugin Settings UI:

| Setting | Default | Description |
|---------|---------|-------------|
| `audio_transcription_enabled` | `true` | Enable/disable audio file transcription |
| `youtube_transcription_enabled` | `true` | Enable/disable YouTube URL transcription |
| `url_audio_transcription_enabled` | `true` | Enable/disable audio URL MIME-type scanning |
| `youtube_max_duration` | `3600` | Max YouTube video length in seconds (0 = no limit) |
| `url_audio_max_size` | `50` | Max audio URL download size in MB |
| `audio_extensions` | see below | List of audio file extensions to detect |
| `transcription_label` | `[Transcription]` | Label prefix for transcription blocks |
| `include_filename` | `true` | Include filename in transcription header |
| `model_bypass_enabled` | `true` | Skip audio transcription for audio-capable models |
| `audio_capable_models` | `gemini-*`, `gpt-4o-audio-*`, ... | Model name wildcard patterns |

## Installation

1. Place the `a0_transcribbler` folder in `usr/plugins/`.
2. Enable the plugin in the Agent Zero Plugins UI.
3. The plugin will auto-install `yt-dlp` on first activation.

## Example Output

When a user sends a voice message saying "Remind me to buy groceries", the LLM
receives:

```
[Transcription] (voice_abc123.ogg):
Remind me to buy groceries

---

[Telegram message from user]
[Voice message — see attachment]
```

## Version

2.0.0
