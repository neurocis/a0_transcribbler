"""A0-Transcribler install hooks.

Installs yt-dlp into the framework runtime for YouTube audio extraction.
Whisper and ffmpeg are already available in the base Agent Zero image.
"""

import subprocess
import sys

from helpers.print_style import PrintStyle


def install():
    """Called by the plugin installer after the plugin is placed."""
    PrintStyle.info("A0-Transcribler: checking dependencies...")

    # Install yt-dlp into the framework venv
    try:
        import yt_dlp  # noqa: F401
        PrintStyle.success("A0-Transcribler: yt-dlp already installed")
    except ImportError:
        PrintStyle.info("A0-Transcribler: installing yt-dlp...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "yt-dlp"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            PrintStyle.success("A0-Transcribler: yt-dlp installed successfully")
        else:
            PrintStyle.warning(
                f"A0-Transcribler: yt-dlp install failed (YouTube transcription "
                f"will be unavailable): {result.stderr[:200]}"
            )

    # Also install yt-dlp as a system command if not present
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=5)
        PrintStyle.success("A0-Transcribler: yt-dlp CLI available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        PrintStyle.info("A0-Transcribler: installing yt-dlp CLI via pip...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "yt-dlp"],
            capture_output=True, text=True
        )

    PrintStyle.success("A0-Transcribler: dependency check complete")
    return 0
