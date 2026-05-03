"""
Validación básica de datos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_file(file_path: Path) -> None:
    """
    Lanza un error claro si un archivo requerido no existe.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo requerido no encontrado: {file_path.as_posix()}"
        )


def require_columns(
    table: pd.DataFrame, required: Iterable[str], table_name: str
) -> None:
    """
    Valida que un DataFrame contenga las columnas esperadas.
    """
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {table_name}: {missing}")
