"""Shared logging utilities for the video understanding agent."""

import sys
import logging

from rich.console import Console
from rich.logging import RichHandler

# Detect if stdout is a real TTY (not redirected to file)
IS_TTY = sys.stdout.isatty()

console = Console(force_terminal=IS_TTY, no_color=not IS_TTY)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with RichHandler (TTY) or StreamHandler (non-TTY)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        if IS_TTY:
            handler = RichHandler(show_path=False, show_time=True, console=console)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger
