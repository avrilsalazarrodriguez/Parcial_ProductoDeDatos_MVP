import pandas as pd

from modelops.data_quality_dimensions import clean_recency_columns, enrich_with_dimensions


def test_recency_99_gets_label():
    df = clean_recency_columns(pd.DataFrame({"recency": [1, 99, None]}))
    assert df.loc[0, "recency_label"] == "1 meses desde última venta"
    assert df.loc[1, "recency_is_missing_or_very_old"] == 1
    assert df.loc[2, "recency_is_missing_or_very_old"] == 1


def test_enrich_keeps_unknown_friendly():
    result = enrich_with_dimensions(pd.DataFrame({"shop_id": [1], "item_id": [10], "segment_name": [None]}))
    assert "segment_name" in result.columns
    assert result.loc[0, "segment_name"] == "Sin clasificar"
