"""A0-Transcribbler core transcription logic.

Reuses Agent Zero's built-in Whisper STT engine for audio file transcription
and yt-dlp for YouTube audio extraction.
"""

import base64
import os
import re
import tempfile
import subprocess
from typing import Optional

from helpers import files, settings, whisper as whisper_helper
from helpers.print_style import PrintStyle

# Audio file extensions considered transcribable
DEFAULT_AUDIO_EXTENSIONS = {
    ".ogg", ".oga", ".mp3", ".wav", ".m4a",
    ".opus", ".flac", ".aac", ".wma", ".webm",
}

# YouTube URL patterns — each captures the video ID in group(1)
YOUTUBE_PATTERNS = [
    re.compile(r'(?:https?://)?(?:www\.)?youtube\.com/watch\?[^\s]*v=([\w-]+)', re.IGNORECASE),
    re.compile(r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([\w-]+)', re.IGNORECASE),
    re.compile(r'(?:https?://)?youtu\.be/([\w-]+)', re.IGNORECASE),
    re.compile(r'(?:https?://)?(?:www\.)?youtube\.com/live/([\w-]+)', re.IGNORECASE),
    re.compile(r'(?:https?://)?music\.youtube\.com/watch\?[^\s]*v=([\w-]+)', re.IGNORECASE),
]


def is_audio_file(filepath: str, audio_extensions: set[str] | None = None) -> bool:
    """Check if a filepath has an audio file extension."""
    exts = audio_extensions or DEFAULT_AUDIO_EXTENSIONS
    _, ext = os.path.splitext(filepath.lower())
    return ext in exts


def extract_youtube_urls(text: str) -> list[str]:
    """Extract YouTube URLs from message text. Returns unique URLs (by video ID)."""
    seen_ids: set[str] = set()
    urls: list[str] = []
    for pattern in YOUTUBE_PATTERNS:
        for match in pattern.finditer(text):
            video_id = match.group(1)
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            full_url = match.group(0)
            if not full_url.startswith("http"):
                full_url = "https://" + full_url
            urls.append(full_url)
    return urls


async def transcribe_audio_file(filepath: str) -> Optional[str]:
    """Transcribe an audio file using Agent Zero's Whisper STT.

    Converts the file to WAV first (via ffmpeg) for maximum compatibility,
    then passes base64-encoded bytes to the Whisper helper.

    Returns the transcription text or None on failure.
    """
    try:
        local_path = files.fix_dev_path(filepath) if hasattr(files, 'fix_dev_path') else filepath

        if not os.path.isfile(local_path):
            PrintStyle.warning(f"A0-Transcribbler: audio file not found: {local_path}")
            return None

        # Convert to WAV using ffmpeg for universal compatibility
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", local_path, "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", wav_path],
                capture_output=True, timeout=120
            )
            if result.returncode != 0:
                PrintStyle.warning(
                    f"A0-Transcribbler: ffmpeg conversion failed for {filepath}: "
                    f"{result.stderr.decode('utf-8', errors='replace')[:200]}"
                )
                return None

            # Read WAV bytes and encode to base64
            with open(wav_path, "rb") as f:
                audio_bytes_b64 = base64.b64encode(f.read()).decode("utf-8")

        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        # Get STT model size from settings
        stt_settings = settings.get_settings()
        model_size = stt_settings.get("stt_model_size", "base")

        # Transcribe using Whisper
        PrintStyle.info(f"A0-Transcribbler: transcribing {os.path.basename(filepath)}...")
        transcription_result = await whisper_helper.transcribe(model_size, audio_bytes_b64)

        if transcription_result and "text" in transcription_result:
            text = transcription_result["text"].strip()
            if text:
                PrintStyle.success(
                    f"A0-Transcribbler: transcribed {os.path.basename(filepath)} "
                    f"({len(text)} chars)"
                )
                return text

        PrintStyle.warning(f"A0-Transcribbler: empty transcription for {filepath}")
        return None

    except Exception as e:
        PrintStyle.error(f"A0-Transcribbler: transcription error for {filepath}: {e}")
        return None


async def transcribe_youtube_url(
    url: str,
    max_duration: int = 3600,
) -> Optional[str]:
    """Download audio from a YouTube URL and transcribe it.

    Uses yt-dlp to download audio, converts to WAV, and transcribes via Whisper.
    Returns transcription text or None on failure.
    """
    try:
        # Check if yt-dlp is available
        yt_dlp_check = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, timeout=10
        )
        if yt_dlp_check.returncode != 0:
            PrintStyle.warning("A0-Transcribbler: yt-dlp not available")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        PrintStyle.warning("A0-Transcribbler: yt-dlp not installed or not responding")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="a0_transcribbler_yt_")
    audio_path = os.path.join(tmp_dir, "audio")
    wav_path = os.path.join(tmp_dir, "audio.wav")

    try:
        # Get video info first to check duration
        if max_duration > 0:
            info_result = subprocess.run(
                ["yt-dlp", "--no-download", "--print", "duration", url],
                capture_output=True, text=True, timeout=30
            )
            if info_result.returncode == 0:
                try:
                    duration = float(info_result.stdout.strip())
                    if duration > max_duration:
                        PrintStyle.warning(
                            f"A0-Transcribbler: YouTube video too long "
                            f"({duration:.0f}s > {max_duration}s limit): {url}"
                        )
                        return None
                except (ValueError, TypeError):
                    pass  # Can't parse duration, proceed anyway

        PrintStyle.info(f"A0-Transcribbler: downloading YouTube audio: {url}")

        # Download audio only
        dl_result = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-x",                      # extract audio
                "--audio-format", "wav",    # convert to wav
                "--audio-quality", "0",     # best quality
                "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
                "-o", audio_path + ".%(ext)s",
                url,
            ],
            capture_output=True, text=True, timeout=300
        )

        if dl_result.returncode != 0:
            PrintStyle.warning(
                f"A0-Transcribbler: yt-dlp download failed for {url}: "
                f"{dl_result.stderr[:200]}"
            )
            return None

        # Find the downloaded file
        downloaded = None
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            if os.path.isfile(fpath) and fname.startswith("audio"):
                downloaded = fpath
                break

        if not downloaded:
            PrintStyle.warning(f"A0-Transcribbler: no downloaded file found for {url}")
            return None

        # If not already WAV, convert
        if not downloaded.endswith(".wav"):
            convert_result = subprocess.run(
                ["ffmpeg", "-y", "-i", downloaded, "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", wav_path],
                capture_output=True, timeout=120
            )
            if convert_result.returncode != 0:
                PrintStyle.warning(f"A0-Transcribbler: ffmpeg conversion failed for YouTube audio")
                return None
            final_path = wav_path
        else:
            final_path = downloaded

        # Read and encode
        with open(final_path, "rb") as f:
            audio_bytes_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Transcribe
        stt_settings = settings.get_settings()
        model_size = stt_settings.get("stt_model_size", "base")

        PrintStyle.info(f"A0-Transcribbler: transcribing YouTube audio from {url}...")
        transcription_result = await whisper_helper.transcribe(model_size, audio_bytes_b64)

        if transcription_result and "text" in transcription_result:
            text = transcription_result["text"].strip()
            if text:
                PrintStyle.success(
                    f"A0-Transcribbler: transcribed YouTube video "
                    f"({len(text)} chars)"
                )
                return text

        PrintStyle.warning(f"A0-Transcribbler: empty transcription for YouTube: {url}")
        return None

    except subprocess.TimeoutExpired:
        PrintStyle.error(f"A0-Transcribbler: timeout processing YouTube URL: {url}")
        return None
    except Exception as e:
        PrintStyle.error(f"A0-Transcribbler: YouTube transcription error: {e}")
        return None
    finally:
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
