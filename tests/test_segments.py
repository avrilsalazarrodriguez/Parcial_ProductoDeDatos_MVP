import pandas as pd

from modelops.segments import build_operational_segments, normalize_categories


def test_build_operational_segments_with_categories():
    monthly = pd.DataFrame(
        {
            "date_block_num": [0, 1, 2, 0, 1],
            "item_id": [1, 1, 1, 2, 2],
            "target": [1.0, 2.0, 0.0, 10.0, 11.0],
            "price_mean": [100.0, 110.0, 100.0, 500.0, 520.0],
        }
    )
    items = pd.DataFrame(
        {
            "item_name": ["a", "b"],
            "item_id": [1, 2],
            "item_category_id": [10, 11],
        }
    )
    categories = pd.DataFrame(
        {
            "item_category_name": ["Games - PS4", "Accessories - PS4"],
            "item_category_id": [10, 11],
        }
    )
    out = build_operational_segments(monthly, items_df=items, categories_df=categories)
    assert "segment_key" in out.columns
    assert out["segment_source"].eq("item_category_id").all()
    assert set(out["segment_key"]) == {"cat_10", "cat_11"}


def test_build_operational_segments_without_items():
    monthly = pd.DataFrame(
        {
            "date_block_num": [0, 1, 2, 0, 1],
            "item_id": [1, 1, 1, 2, 2],
            "target": [1.0, 2.0, 0.0, 10.0, 11.0],
            "price_mean": [100.0, 110.0, 100.0, 500.0, 520.0],
        }
    )
    out = build_operational_segments(monthly, items_df=None, categories_df=None)
    assert "segment_key" in out.columns
    assert out["item_id"].nunique() == 2
    assert out["segment_source"].eq("derived_operational_group").all()


def test_normalize_categories_extracts_group():
    categories = pd.DataFrame(
        {
            "item_category_name": ["PC - Headsets / Headphones"],
            "item_category_id": [0],
        }
    )
    out = normalize_categories(categories)
    assert out.loc[0, "category_group"] == "PC"
