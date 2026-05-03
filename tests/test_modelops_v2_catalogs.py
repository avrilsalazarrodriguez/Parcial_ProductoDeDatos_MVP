import pandas as pd

from src.modelops_v2.catalogs import add_catalog_metadata


def test_catalog_fallback_names_do_not_show_unknown(tmp_path):
    df = pd.DataFrame({"shop_id": [2], "item_id": [5037], "prediction": [1.2], "recency": [99]})
    enriched = add_catalog_metadata(df, data_dir=tmp_path)
    assert enriched.loc[0, "shop_name"] == "Tienda 2"
    assert enriched.loc[0, "item_name"] == "Producto 5037"
    assert "Sin ventas" in enriched.loc[0, "recency_display"]
