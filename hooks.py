"""A0-Transcribbler install hooks.

Installs yt-dlp into the framework runtime for YouTube audio extraction.
Whisper and ffmpeg are already available in the base Agent Zero image.

Creates a status file to track dependency availability for runtime checks.
"""

import subprocess
import sys
import os
import json
from datetime import datetime

from helpers.print_style import PrintStyle

# Plugin directory (where this file lives)
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(PLUGIN_DIR, ".dependency_status.json")


def _write_status(status: dict) -> None:
    """Write dependency status to a JSON file for runtime checks."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        PrintStyle.warning(f"A0-Transcribbler: could not write status file: {e}")


def _check_yt_dlp_module() -> bool:
    """Check if yt-dlp Python module is importable."""
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def _check_yt_dlp_cli() -> bool:
    """Check if yt-dlp CLI is available."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install():
    """Called by the plugin installer after the plugin is placed.
    
    Ensures yt-dlp is installed and creates a status file for runtime checks.
    Returns 0 on success, non-zero on critical failure.
    """
    PrintStyle.info("A0-Transcribbler: checking dependencies...")

    status = {
        "checked_at": datetime.now().isoformat(),
        "yt_dlp_module": False,
        "yt_dlp_cli": False,
        "warnings": [],
        "errors": [],
    }

    # --- Check and install yt-dlp Python module ---
    if _check_yt_dlp_module():
        PrintStyle.success("A0-Transcribbler: yt-dlp module already installed")
        status["yt_dlp_module"] = True
    else:
        PrintStyle.info("A0-Transcribbler: installing yt-dlp module...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "yt-dlp"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and _check_yt_dlp_module():
            PrintStyle.success("A0-Transcribbler: yt-dlp module installed successfully")
            status["yt_dlp_module"] = True
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown installation error"
            PrintStyle.error(
                f"A0-Transcribbler: yt-dlp module installation failed: {error_msg}"
            )
            status["errors"].append(f"yt-dlp module installation failed: {error_msg}")
            # Continue - YouTube transcription will be unavailable but audio files still work

    # --- Check and install yt-dlp CLI ---
    if _check_yt_dlp_cli():
        PrintStyle.success("A0-Transcribbler: yt-dlp CLI available")
        status["yt_dlp_cli"] = True
    else:
        # CLI might be available via the module even if not on PATH
        # Try installing again to ensure CLI is available
        PrintStyle.info("A0-Transcribbler: ensuring yt-dlp CLI is available...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "yt-dlp"],
            capture_output=True, text=True, timeout=120
        )
        if _check_yt_dlp_cli():
            PrintStyle.success("A0-Transcribbler: yt-dlp CLI now available")
            status["yt_dlp_cli"] = True
        else:
            # Not critical - the module can be used directly
            PrintStyle.warning(
                "A0-Transcribbler: yt-dlp CLI not on PATH; "
                "YouTube transcription will use Python module fallback"
            )
            status["warnings"].append("yt-dlp CLI not on PATH")
            # Still mark as working if module is available
            status["yt_dlp_cli"] = status["yt_dlp_module"]

    # --- Summary ---
    if status["errors"]:
        PrintStyle.error(
            f"A0-Transcribbler: initialization completed with errors. "
            f"YouTube transcription may be unavailable. Check status file for details."
        )
    elif status["warnings"]:
        PrintStyle.warning(
            f"A0-Transcribbler: initialization completed with warnings. "
            f"See status file for details."
        )
    else:
        PrintStyle.success("A0-Transcribbler: dependency check complete")

    # Write status file for runtime checks
    _write_status(status)

    # Return 0 even on partial failure to allow plugin to load
    # Runtime code will check status file and handle accordingly
    return 0


def check_yt_dlp_available() -> bool:
    """Runtime check for yt-dlp availability.
    
    Call this from the extension hook to verify yt-dlp is working
    before attempting YouTube transcription.
    
    Returns True if yt-dlp is available, False otherwise.
    """
    # First check status file from installation
    if os.path.isfile(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                status = json.load(f)
                if status.get("yt_dlp_module") or status.get("yt_dlp_cli"):
                    return True
        except Exception:
            pass

    # Fallback to runtime check
    return _check_yt_dlp_module() or _check_yt_dlp_cli()
