import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(base_dir: Path, level=logging.INFO, log_name: str = "pipeline.log"):
    """
    Sätter upp logging med både konsol- och filoutput.
    Loggar alltid till en undermapp 'logs/' (standard: pipeline.log).

    - Konsolen visar endast INFO och uppåt (för överskådlighet).
    - Loggfilen tar emot alla nivåer (DEBUG och uppåt) och roteras
      automatiskt om den blir för stor.

    Parametrar
    ----------
    base_dir : Path
        Basmapp där 'logs/' skapas.
    level : int, default=logging.INFO
        Lägsta nivå som visas i konsolen.
    log_name : str, default="pipeline.log"
        Namn på loggfilen.
    """
    # Skapa loggmapp om den inte redan finns
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Filväg för loggfil
    log_file = log_dir / log_name

    # Format på loggrader
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Konsol (bara INFO och uppåt)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    # Loggfil (DEBUG och uppåt, roterande fil max 2 MB, max 3 backuper)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Rensa ev. gamla handlers (för att undvika dubbletter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Aktivera båda handlers: konsol + fil
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initierad → %s", log_file)
    return root_logger
