from src.modelops_v2.metrics import evaluate_predictions


def test_model_beats_naive_wape():
    y = [0, 1, 2, 10]
    pred = [0, 1, 2, 8]
    naive = [0, 0, 0, 0]
    metrics = evaluate_predictions(y, pred, naive)
    assert metrics.beats_naive_wape is True
    assert metrics.wape < 1


def test_zero_series_does_not_crash():
    metrics = evaluate_predictions([0, 0, 0], [0, 0, 0], [0, 0, 0])
    assert metrics.wape == 0
    assert metrics.mae == 0
