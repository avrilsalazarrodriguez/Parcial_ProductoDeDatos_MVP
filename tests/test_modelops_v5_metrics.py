from src.modelops_v3.metrics import compute_metrics, ModelRunResult, select_champion, results_df


def test_compute_metrics_has_rmse_and_mae():
    metrics = compute_metrics([1, 2, 3], [1, 1, 5])
    assert "rmse" in metrics
    assert "mae" in metrics


def test_select_champion_uses_rmse_then_mae():
    a = ModelRunResult("a", "A", "fam", rmse=0.8, mae=0.5, smape=0.2, wape=0.2, bias=0, nonzero_recall=1, n_valid=10)
    b = ModelRunResult("b", "B", "fam", rmse=0.9, mae=0.1, smape=0.2, wape=0.2, bias=0, nonzero_recall=1, n_valid=10)
    out = select_champion([a, b])
    champion = [x for x in out if x.is_champion][0]
    assert champion.model_id == "a"


def test_results_df_champion_first():
    a = ModelRunResult("a", "A", "fam", rmse=0.8, mae=0.5, smape=0.2, wape=0.2, bias=0, nonzero_recall=1, n_valid=10, is_champion=True)
    b = ModelRunResult("b", "B", "fam", rmse=0.9, mae=0.1, smape=0.2, wape=0.2, bias=0, nonzero_recall=1, n_valid=10)
    df = results_df([b, a])
    assert df.iloc[0]["model_id"] == "a"
