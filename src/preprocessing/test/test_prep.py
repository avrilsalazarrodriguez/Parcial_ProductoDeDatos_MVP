import pytest
import pandas as pd

from src.preprocessing.prep import construir_target_mensual


# TEST 1 - Agregación correcta de datos

def test_construir_target_mensual_agrega_correctamente():
    """
    Este test verifica que la función construir_target_mensual:
    
    1. Agrupe correctamente por (mes, tienda, producto)
    2. Sume correctamente las ventas diarias
    """

    # Creamos un DataFrame pequeño artificial
    # Simulamos dos ventas del mismo producto
    df = pd.DataFrame(
        {
            # Mismo mes
            "date_block_num": [0, 0],
            # Misma tienda
            "shop_id": [1, 1],
            # Mismo producto
            "item_id": [10, 10],
            # Precio
            "item_price": [100.0, 100.0],
            # Ventas diarias
            "item_cnt_day": [2, 3],
        }
    )

    # Ejecutamos la función que queremos probar
    result = construir_target_mensual(df)

    # 1. Verificamos que después del groupby solo quede una fila
    # porque todas pertenecen al mismo mes, tienda y producto
    assert result.shape[0] == 1

    # 2. Verificamos que la suma mensual sea correcta (2 + 3 = 5)
    assert result["item_cnt_month"].iloc[0] == 5


# TEST 2 - Error si falta la columna item_cnt_day

def test_construir_target_mensual_falla_si_falta_columna():
    """
    Este test verifica que la función falle si falta una columna obligatoria.
    
    Caso probado:
    - Eliminamos 'item_cnt_day', que es esencial para calcular el target mensual.
    - Esperamos que la función lance un error.
    
    Esto protege contra errores de implementación donde el dataset
    llegue incompleto o mal procesado.
    """

    # Creamos DataFrame SIN la columna 'item_cnt_day'
    df = pd.DataFrame(
        {
            "date_block_num": [0, 0],
            "shop_id": [1, 1],
            "item_id": [10, 10],
            "item_price": [100.0, 100.0],
            # Intencionalmente quitamos item_cnt_day
        }
    )

    # pytest.raises verifica que la función lance una excepción
    with pytest.raises(KeyError):
        construir_target_mensual(df)


# TEST 3 - Validación de tipo numérico en item_cnt_month

def test_construir_target_mensual_output_es_numerico():
    """
    Este test verifica que la columna 'item_cnt_month'
    generada por la función sea numérica.
    
    Esto es importante porque:
    - El modelo espera valores numéricos
    - Evita errores silenciosos si el dtype cambia
    """

    # DataFrame de ejemplo válido
    df = pd.DataFrame(
        {
            "date_block_num": [0, 0],
            "shop_id": [1, 1],
            "item_id": [10, 10],
            "item_price": [100.0, 100.0],
            "item_cnt_day": [2, 3],
        }
    )

    result = construir_target_mensual(df)

    # Verificamos que la columna exista
    assert "item_cnt_month" in result.columns

    # Verificamos que el tipo sea numérico
    assert pd.api.types.is_numeric_dtype(result["item_cnt_month"])


# TEST 4 - Clipping de outliers

def test_construir_target_mensual_aplica_clipping():
    """
    Este test verifica que si la suma mensual excede TARGET_MAX,
    el valor sea recortado correctamente.
    
    Esto protege contra outliers extremos que podrían
    afectar el entrenamiento del modelo.
    """

    # Creamos ventas altas
    df = pd.DataFrame(
        {
            "date_block_num": [0, 0],
            "shop_id": [1, 1],
            "item_id": [10, 10],
            "item_price": [100.0, 100.0],
            "item_cnt_day": [15, 15],  # 15 + 15 = 30
        }
    )

    result = construir_target_mensual(df)

    # Como TARGET_MAX es 20 en la config,
    # el resultado debería ser 20, no 30.
    assert result["item_cnt_month"].iloc[0] <= 20