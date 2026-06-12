from mlscorepredictor.data.loaders import DataCatalog
from mlscorepredictor.modeling.evaluation import evaluate_historical_predictions
from mlscorepredictor.modeling.predictor import FootballPredictor
from mlscorepredictor.modeling.simulation import expected_group_table, simulate_group_stage


def test_group_stage_simulation_returns_all_teams() -> None:
    catalog = DataCatalog()
    predictor = FootballPredictor(catalog=catalog)

    simulation = simulate_group_stage(predictor, catalog.fixtures(), iterations=20, seed=7)

    assert len(simulation) == 48
    assert simulation["advance"].between(0, 1).all()


def test_expected_group_table_returns_four_teams() -> None:
    catalog = DataCatalog()
    predictor = FootballPredictor(catalog=catalog)
    group_a = catalog.fixtures()[catalog.fixtures()["group"].eq("A")]

    table = expected_group_table(predictor, group_a)

    assert len(table) == 4
    assert table["expected_points"].is_monotonic_decreasing


def test_historical_evaluation_returns_metrics() -> None:
    metrics = evaluate_historical_predictions(FootballPredictor())

    assert metrics["matches"] > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["log_loss"] > 0
