"""Consistent terminal presentation for the MLB analyst.

The predictor is a command-line application, so it cannot reliably control
the font size of every terminal emulator. On Windows legacy console hosts we
make a best-effort attempt to scale the current console font by 1.4x. Modern
Windows Terminal profiles may ignore that API; in that case the colour theme
still works and the user can set the profile font size manually.

Visible ANSI output deliberately uses only bright white and purple. The
helpers are no-ops when output is redirected, or when ``NO_COLOR`` /
``PREDICTOR_COLOR=0`` is set, so logs remain clean text.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
from ctypes import wintypes
from typing import TextIO


ANSI_RESET = "\033[0m"
ANSI_WHITE = "\033[97m"
ANSI_PURPLE = "\033[95m"
ANSI_BOLD_WHITE = "\033[1;97m"
ANSI_DIM_WHITE = ANSI_WHITE

DEFAULT_FONT_SCALE = 1.40

_COLOR_ENABLED = False
_CONFIGURED = False


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def colors_enabled() -> bool:
    """Return whether ANSI presentation is enabled for this process."""

    return _COLOR_ENABLED


def set_color_enabled(enabled: bool) -> None:
    """Set colour state explicitly; useful for tests and embedded callers."""

    global _COLOR_ENABLED
    _COLOR_ENABLED = bool(enabled)


def _styled(text: object, prefix: str) -> str:
    value = str(text)
    if not _COLOR_ENABLED:
        return value
    # Restore white after every segment so unwrapped text and nested purple
    # values do not fall back to the terminal's arbitrary default colour.
    return f"{prefix}{value}{ANSI_RESET}{ANSI_WHITE}"


def white(text: object) -> str:
    return _styled(text, ANSI_WHITE)


def bold(text: object) -> str:
    return _styled(text, ANSI_BOLD_WHITE)


def dim(text: object) -> str:
    # Keep secondary detail white as well. Hierarchy comes from layout and
    # boldness, not a third grey colour.
    return _styled(text, ANSI_DIM_WHITE)


def purple(text: object) -> str:
    return _styled(text, ANSI_PURPLE)


def _requested_font_scale() -> float | None:
    raw = os.getenv("PREDICTOR_FONT_SCALE", str(DEFAULT_FONT_SCALE)).strip()
    if raw.lower() in {"0", "false", "no", "off", "none"}:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = DEFAULT_FONT_SCALE
    # Avoid accidental giant font changes from a typo in an environment var.
    return max(0.75, min(3.0, value))


def _font_marker_name(scale: float) -> str:
    return f"MLBAnalystFontScale{int(round(scale * 100)):03d}"


def _try_resize_windows_console_font(scale: float, stream: TextIO) -> bool:
    """Best-effort idempotent resize for a classic Windows console host."""

    if os.name != "nt" or not _is_tty(stream):
        return False
    if not _env_flag("PREDICTOR_RESIZE_FONT", True) or scale <= 1.0:
        return False

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        class _Coord(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class _ConsoleFontInfoEx(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("nFont", wintypes.DWORD),
                ("dwFontSize", _Coord),
                ("FontFamily", wintypes.UINT),
                ("FontWeight", wintypes.UINT),
                ("FaceName", wintypes.WCHAR * 32),
            ]

        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetCurrentConsoleFontEx.argtypes = [
            wintypes.HANDLE,
            wintypes.BOOL,
            ctypes.POINTER(_ConsoleFontInfoEx),
        ]
        kernel32.GetCurrentConsoleFontEx.restype = wintypes.BOOL
        kernel32.SetCurrentConsoleFontEx.argtypes = [
            wintypes.HANDLE,
            wintypes.BOOL,
            ctypes.POINTER(_ConsoleFontInfoEx),
        ]
        kernel32.SetCurrentConsoleFontEx.restype = wintypes.BOOL
        kernel32.GetConsoleWindow.restype = wintypes.HWND

        user32.GetPropW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        user32.GetPropW.restype = wintypes.HANDLE
        user32.SetPropW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.HANDLE]
        user32.SetPropW.restype = wintypes.BOOL

        hwnd = kernel32.GetConsoleWindow()
        marker = _font_marker_name(scale)
        if hwnd and user32.GetPropW(hwnd, marker):
            # The console window has already been adjusted during this
            # session. Do not compound 1.4x on every predictor invocation.
            return True

        # STD_OUTPUT_HANDLE = -11. Use the current output console font, not a
        # hard-coded font face or absolute size, so the user's typeface stays.
        handle = kernel32.GetStdHandle(ctypes.c_ulong(-11).value)
        invalid_handle = ctypes.c_void_p(-1).value
        if not handle or handle == invalid_handle:
            return False

        info = _ConsoleFontInfoEx()
        info.cbSize = ctypes.sizeof(_ConsoleFontInfoEx)
        if not kernel32.GetCurrentConsoleFontEx(handle, False, ctypes.byref(info)):
            return False

        current_x = max(1, int(info.dwFontSize.X))
        current_y = max(1, int(info.dwFontSize.Y))
        info.dwFontSize.X = max(1, min(72, int(round(current_x * scale))))
        info.dwFontSize.Y = max(1, min(72, int(round(current_y * scale))))
        if not kernel32.SetCurrentConsoleFontEx(handle, False, ctypes.byref(info)):
            return False

        if hwnd:
            # A window property lasts for the life of the console window and
            # prevents repeated runs from multiplying the font size.
            user32.SetPropW(hwnd, marker, wintypes.HANDLE(1))
        return True
    except (AttributeError, OSError, ctypes.ArgumentError, TypeError):
        # Windows Terminal/ConPTY and non-Windows hosts may not expose the
        # classic console API. Styling must never make prediction fail.
        return False


def configure_terminal_display(stream: TextIO | None = None) -> dict[str, object]:
    """Enable the white/purple theme and try to enlarge a Windows console.

    ``PREDICTOR_FONT_SCALE=1.0`` or ``PREDICTOR_RESIZE_FONT=0`` disables the
    font adjustment. ``PREDICTOR_COLOR=0`` or ``NO_COLOR`` disables ANSI
    colours. Colour output is automatically disabled when stdout is not a
    TTY, which keeps redirected reports and logs readable.
    """

    global _COLOR_ENABLED, _CONFIGURED
    stream = stream or sys.stdout
    if not _CONFIGURED:
        color_override = os.getenv("PREDICTOR_COLOR")
        color_requested = (
            _is_tty(stream)
            if color_override is None
            else _env_flag("PREDICTOR_COLOR", True)
        )
        _COLOR_ENABLED = "NO_COLOR" not in os.environ and color_requested
        _CONFIGURED = True
        if _COLOR_ENABLED:
            try:
                stream.write(ANSI_WHITE)
                stream.flush()
            except (AttributeError, OSError):
                pass
        font_scale = _requested_font_scale()
        font_resized = (
            _try_resize_windows_console_font(font_scale, stream)
            if font_scale is not None
            else False
        )
    else:
        font_resized = False

    return {
        "colors_enabled": _COLOR_ENABLED,
        "font_scale": _requested_font_scale(),
        "font_resized": font_resized,
    }


def reset_terminal_display(stream: TextIO | None = None) -> None:
    """Restore the terminal's normal style when the process exits."""

    if not _COLOR_ENABLED:
        return
    stream = stream or sys.stdout
    try:
        stream.write(ANSI_RESET)
        stream.flush()
    except (AttributeError, OSError):
        pass


atexit.register(reset_terminal_display)
