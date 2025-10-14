"""
Central loggning:

- färgade konsolloggar
- loggfiler i logs/-mapp
- rensning av dubblett-handlers
- global standard för logging
"""


from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime


class LogColors:
    """ANSI-färger för terminal-loggar."""
    RESET = "\033[0m"
    GREY = "\033[90m"       # DEBUG
    YELLOW = "\033[93m"     # WARNING
    RED = "\033[91m"        # ERROR
    MAGENTA = "\033[95m"    # CRITICAL
    CYAN = "\033[96m"       # INFO


class ColorFormatter(logging.Formatter):
    """Formatter som färgar loggar baserat på nivå."""
    COLORS = {
        "DEBUG": LogColors.GREY,
        "INFO": LogColors.CYAN,
        "WARNING": LogColors.YELLOW,
        "ERROR": LogColors.RED,
        "CRITICAL": LogColors.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, LogColors.RESET)
        message = record.getMessage()
        return f"{color}{record.levelname:<7} {message}{LogColors.RESET}"


def _project_root() -> Path:
    """Returnerar projektets rotmapp."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "src").exists() or (parent / ".git").exists():
            return parent
    return here.parent


def setup_logging(level: int = logging.INFO) -> Path:
    """Initierar loggning och skapar loggfil."""
    project_root = _project_root()
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Rensa gamla handlers för att undvika dubbletter
    logging.getLogger().handlers.clear()

    # Konsollogg med färg
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    console_handler.setLevel(level)

    # Loggfil
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Aktivera logging globalt
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    return log_file