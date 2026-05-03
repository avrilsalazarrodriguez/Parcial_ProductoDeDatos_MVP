"""
Utilidades de logging.

Provee una configuración consistente de logger (archivo + consola) para todos los scripts.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

FORMATO_LOG = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(nombre: str, log_dir: Path, prefijo_archivo: str) -> logging.Logger:
    """
    Crea (o reutiliza) un logger configurado.

    Parámetros
    ----------
    nombre:
        Nombre del logger (normalmente __name__).
    log_dir:
        Directorio donde se guardarán los logs.
    prefijo_archivo:
        Prefijo para el nombre del archivo de log (se le añadirá un timestamp).

    Regresa
    -------
    logging.Logger
        Logger configurado para escribir en consola y archivo.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(nombre)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        # Ya está configurado; no agregamos handlers duplicados.
        return logger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_log = log_dir / f"{prefijo_archivo}_{timestamp}.log"

    file_handler = logging.FileHandler(ruta_log, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(FORMATO_LOG))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(FORMATO_LOG))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
