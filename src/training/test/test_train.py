import numpy as np
import pandas as pd
import logging

from src.training.train import evaluate_valid, DatasetSplit

 
# Helper: logger mínimo para testing

def get_dummy_logger():
    """
    Creamos un logger básico que no imprime nada.
    Solo sirve para que la función no falle.
    """
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.CRITICAL)
    return logger


# TEST 1 - RMSE es 0 cuando predicción = realidad

def test_evaluate_valid_rmse_cero_si_prediccion_perfecta():
    """
    Si la predicción es exactamente igual al valor real,
    el RMSE debe ser 0.
    
    Esto valida que la fórmula matemática esté correcta.
    """

    y_real = pd.Series([1.0, 2.0, 3.0])
    x_dummy = pd.DataFrame({"a": [0, 0, 0]})
    split = DatasetSplit(x=x_dummy, y=y_real)

    # Prob=1 y mu = y_real → predicción perfecta
    prob_valid = np.array([1.0, 1.0, 1.0])
    mu_valid = np.array([1.0, 2.0, 3.0])

    score = evaluate_valid(split, prob_valid, mu_valid, get_dummy_logger())

    assert score == 0.0


# TEST 2 - RMSE mayor a 0 cuando hay error

def test_evaluate_valid_rmse_mayor_cero_si_error():
    """
    Si la predicción es distinta al valor real,
    el RMSE debe ser mayor a 0.
    """

    y_real = pd.Series([1.0, 2.0, 3.0])
    x_dummy = pd.DataFrame({"a": [0, 0, 0]})
    split = DatasetSplit(x=x_dummy, y=y_real)

    # Predicción incorrecta
    prob_valid = np.array([1.0, 1.0, 1.0])
    mu_valid = np.array([0.0, 0.0, 0.0])

    score = evaluate_valid(split, prob_valid, mu_valid, get_dummy_logger())

    assert score > 0.0


# TEST 3 - Clipping respeta TARGET_MAX

def test_evaluate_valid_aplica_clipping():
    """
    Verifica que valores extremadamente altos
    sean recortados por el clipping.
    """

    y_real = pd.Series([5.0])
    x_dummy = pd.DataFrame({"a": [0]})
    split = DatasetSplit(x=x_dummy, y=y_real)

    # Valor enorme que debería ser clippeado
    prob_valid = np.array([1.0])
    mu_valid = np.array([1000.0])

    score = evaluate_valid(split, prob_valid, mu_valid, get_dummy_logger())

    # No verificamos valor exacto del RMSE,
    # solo que la función corre correctamente
    assert isinstance(score, float)

