"""A0-Transcribbler: intercept user messages before LLM processing.

This extension hooks into Agent.hist_add_user_message (start) to detect
audio file attachments and YouTube URLs, transcribe them, and prepend the
transcription text to the user message so the LLM receives it.

If the active chat model natively supports audio input (e.g. Gemini),
audio file transcription is skipped and the attachment passes through
for direct LLM processing. YouTube URLs are always transcribed since
models cannot fetch remote content.

hist_add_user_message is a sync method called from an async context.
Since Whisper transcription is async, we run it in a dedicated thread
with its own event loop.
"""

import asyncio
import fnmatch
import os
import threading

from helpers.extension import Extension
from helpers import plugins
from helpers.print_style import PrintStyle

PLUGIN_NAME = "a0_transcribbler"


def _run_async_in_thread(coro):
    """Run an async coroutine in a new thread with its own event loop.
    Returns the result. Blocks until complete.
    """
    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = asyncio.run(coro)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=300)  # 5 min timeout for long transcriptions

    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0]


def _is_model_audio_capable(agent, config: dict) -> bool:
    """Check if the active chat model natively supports audio input.

    Compares the model name against configurable wildcard patterns.
    Returns True if the model matches any audio-capable pattern.
    """
    if not config.get("model_bypass_enabled", True):
        return False

    patterns = config.get("audio_capable_models", [])
    if not patterns:
        return False

    # Get active chat model name from model config
    try:
        from plugins._model_config.helpers.model_config import get_chat_model_config
        chat_cfg = get_chat_model_config(agent)
        model_name = (chat_cfg.get("name", "") or "").strip().lower()
        if not model_name:
            return False
    except Exception:
        return False

    # Check against each pattern (case-insensitive wildcard matching)
    for pattern in patterns:
        pattern_lower = pattern.strip().lower()
        if fnmatch.fnmatch(model_name, pattern_lower):
            PrintStyle.info(
                f"A0-Transcribbler: model '{model_name}' matches "
                f"audio-capable pattern '{pattern}' — bypassing audio transcription"
            )
            return True

    return False


class TranscribeOnMessage(Extension):

    def execute(self, **kwargs) -> None:
        data = kwargs.get("data")
        if not data:
            return

        args = data.get("args", ())

        # hist_add_user_message(self, message, intervention=False)
        # extensible wraps: args = (self/agent, message, ...)
        if len(args) < 2:
            return

        agent = args[0]
        message = args[1]

        # Quick check: does this message have anything to transcribe?
        has_attachments = hasattr(message, "attachments") and message.attachments
        has_message = hasattr(message, "message") and message.message
        if not has_attachments and not has_message:
            return

        # Get plugin config
        config = plugins.get_plugin_config(PLUGIN_NAME, agent) or {}
        audio_enabled = config.get("audio_transcription_enabled", True)
        youtube_enabled = config.get("youtube_transcription_enabled", True)

        if not audio_enabled and not youtube_enabled:
            return

        # Check if the active model natively supports audio
        model_handles_audio = _is_model_audio_capable(agent, config)

        # If model handles audio, skip audio transcription but still do YouTube
        if model_handles_audio and audio_enabled:
            PrintStyle.info(
                "A0-Transcribbler: audio-capable model detected, "
                "skipping audio file transcription (passthrough)"
            )
            audio_enabled = False  # Bypass audio transcription

        if not audio_enabled and not youtube_enabled:
            return

        # Import transcriber helpers
        from usr.plugins.a0_transcribbler.helpers.transcriber import (
            is_audio_file,
            extract_youtube_urls,
            transcribe_audio_file,
            transcribe_youtube_url,
        )

        transcription_parts = []
        label = config.get("transcription_label", "[Transcription]")
        include_filename = config.get("include_filename", True)

        # Build audio extensions set from config
        audio_exts_list = config.get("audio_extensions", [])
        audio_exts = set(audio_exts_list) if audio_exts_list else None

        # --- 1. Transcribe audio file attachments ---
        if audio_enabled and has_attachments:
            for attachment in message.attachments:
                if not is_audio_file(attachment, audio_exts):
                    continue

                fname = os.path.basename(attachment)
                PrintStyle.info(f"A0-Transcribbler: detected audio: {fname}")

                try:
                    text = _run_async_in_thread(
                        transcribe_audio_file(attachment)
                    )
                    if text:
                        header = label
                        if include_filename:
                            header += f" ({fname})"
                        transcription_parts.append(f"{header}:\n{text}")
                except Exception as e:
                    PrintStyle.error(
                        f"A0-Transcribbler: failed to transcribe {fname}: {e}"
                    )

        # --- 2. Transcribe YouTube URLs found in message ---
        if youtube_enabled and has_message:
            youtube_urls = extract_youtube_urls(message.message)
            max_duration = config.get("youtube_max_duration", 3600)

            for url in youtube_urls:
                PrintStyle.info(f"A0-Transcribbler: detected YouTube: {url}")

                try:
                    text = _run_async_in_thread(
                        transcribe_youtube_url(url, max_duration)
                    )
                    if text:
                        transcription_parts.append(
                            f"{label} (YouTube: {url}):\n{text}"
                        )
                except Exception as e:
                    PrintStyle.error(
                        f"A0-Transcribbler: failed to transcribe YouTube "
                        f"{url}: {e}"
                    )

        # --- 3. Inject transcriptions into message ---
        if transcription_parts:
            transcription_block = "\n\n".join(transcription_parts)
            original_msg = message.message or ""
            message.message = (
                f"{transcription_block}\n\n"
                f"---\n\n"
                f"{original_msg}"
            )
            PrintStyle.success(
                f"A0-Transcribbler: injected {len(transcription_parts)} "
                f"transcription(s) into message"
            )

            # Update args with modified message
            data["args"] = (args[0], message) + args[2:]
