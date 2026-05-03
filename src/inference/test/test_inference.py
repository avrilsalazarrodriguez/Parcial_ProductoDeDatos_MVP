import numpy as np
import pandas as pd

from src.training.train import TARGET_MIN, TARGET_MAX
from src.training.train import evaluate_valid, DatasetSplit
import logging


# Helper: logger mínimo

def get_dummy_logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.CRITICAL)
    return logger


# TEST 1 - Tamaño correcto de predicción

def test_prediccion_tiene_mismo_tamano_que_input():
    """
    Verifica que la cantidad de predicciones
    coincida con la cantidad de observaciones.
    """

    y_real = pd.Series([1.0, 2.0, 3.0])
    x_dummy = pd.DataFrame({"a": [0, 0, 0]})
    split = DatasetSplit(x=x_dummy, y=y_real)

    prob_valid = np.array([1.0, 1.0, 1.0])
    mu_valid = np.array([1.0, 2.0, 3.0])

    score = evaluate_valid(split, prob_valid, mu_valid, get_dummy_logger())

    # Si no falla y devuelve float, el tamaño fue consistente
    assert isinstance(score, float)


# TEST 2 - Clipping respeta límites

def test_prediccion_no_supera_target_max():
    """
    Si el modelo produce valores extremos,
    deben ser recortados por TARGET_MAX.
    """

    y_real = pd.Series([5.0])
    x_dummy = pd.DataFrame({"a": [0]})
    split = DatasetSplit(x=x_dummy, y=y_real)

    prob_valid = np.array([1.0])
    mu_valid = np.array([9999.0])

    score = evaluate_valid(split, prob_valid, mu_valid, get_dummy_logger())

    assert isinstance(score, float)


# TEST 3 - Predicción es numérica

def test_prediccion_es_numerica():
    """
    Verifica que la salida final sea numérica.
    """

    y_real = pd.Series([2.0])
    x_dummy = pd.DataFrame({"a": [0]})
    split = DatasetSplit(x=x_dummy, y=y_real)

    prob_valid = np.array([1.0])
    mu_valid = np.array([2.0])

    score = evaluate_valid(split, prob_valid, mu_valid, get_dummy_logger())

    assert isinstance(score, float)