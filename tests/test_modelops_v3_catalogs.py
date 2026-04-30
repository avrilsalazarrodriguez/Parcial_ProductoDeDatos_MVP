from __future__ import annotations

import pandas as pd

from src.modelops_v3.catalogs import clean_metadata


def test_recency_99_label():
    df = pd.DataFrame({"recency": [99, 4], "item_category_id": [-1, 10]})
    out = clean_metadata(df)
    assert out.loc[0, "recency_label"] == "Sin ventas recientes / sin historial suficiente"
    assert out.loc[1, "recency_label"] == "4 meses"
    assert out.loc[0, "metadata_status"] == "metadata_missing"
