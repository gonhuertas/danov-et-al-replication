"""Validate saved empirical results, source hashes, timing and report assets."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
import unittest
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from data_model import SERIES, forecast_dates, training_window, transform_panel
from file_utils import ROOT, output_session, write_json, write_text

RESULTS = ROOT / "results"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value and not urlsplit(value).scheme and not value.startswith("#"):
                self.paths.append(unquote(value.split("#", 1)[0]))


def validate():
    checks = []
    suite = unittest.defaultTestLoader.loadTestsFromName("test_replication")
    stream = StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write_text(RESULTS / "test_results.txt", stream.getvalue())
    assert tests.wasSuccessful(), stream.getvalue()
    checks.append(f"All {tests.testsRun} focused behavior tests passed.")
    raw = pd.read_csv(ROOT / "data" / "observations.csv")
    assert not raw.duplicated(["series", "period"]).any()
    panel, _ = transform_panel(raw)
    assert "pmi" not in panel and len(panel.columns) == 11
    assert (raw.series == "gdp").sum() == 67
    assert raw.value.notna().all()
    assert len(panel.index) == 192
    for key in panel:
        frequency = "Q" if key == "gdp" else "M"
        periods = pd.PeriodIndex(raw.loc[raw.series == key, "period"], freq=frequency)
        assert periods.max() <= pd.Period(SERIES[key]["end"], freq=frequency)
    checks.append("Input panel has 10 monthly indicators and 67 actual quarterly GDP levels; no duplicate periods or post-paper endpoints.")
    for meta in (ROOT / "data" / "raw").glob("*.meta.json"):
        source = meta.with_name(meta.name.removesuffix(".meta.json"))
        record = json.loads(meta.read_text(encoding="utf-8"))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record["sha256"], source.name
    checks.append("Every cached source document matches its acquisition checksum.")
    predictions = pd.read_csv(RESULTS / "quarterly_forecasts.csv")
    expected_dates = {str(date.date()) for date in forecast_dates()}
    assert len(predictions) == 618
    assert not predictions.duplicated(["asof", "target", "factors"]).any()
    for count, rows in predictions.groupby("factors"):
        assert set(rows["asof"]) == expected_dates and len(rows) == 206
        assert np.isfinite(rows[["prediction", "conditional_sd", "lower95", "upper95"]]).all().all()
        assert (rows.conditional_sd >= 0).all()
        assert (rows.lower95 <= rows.prediction).all() and (rows.prediction <= rows.upper95).all()
        evaluation = rows.loc[rows.target.between("2021Q4", "2025Q3")]
        assert evaluation.target.nunique() == 16 and evaluation.actual_log_yoy.notna().all()
        assert evaluation.groupby("type").size().to_dict() == {"backcast": 96, "nowcast": 96}
    checks.append("All 618 forecasts are finite, uniquely dated and properly bounded; all models score the same 16 completed quarters.")
    news = pd.read_csv(RESULTS / "news_details.csv")
    by_release = news.groupby(["asof", "target", "factors"]).impact.sum()
    indexed = predictions.set_index(["asof", "target", "factors"])
    attributed = by_release.reindex(indexed.index, fill_value=0)
    residual = indexed.release_news_effect - attributed
    assert residual.abs().max() < 1e-7
    for _, group in predictions.groupby(["target", "factors"]):
        group = group.sort_values("asof")
        change = group.prediction.diff().iloc[1:]
        explanation = (group.release_news_effect + group.reestimation_effect).iloc[1:]
        if len(change):
            assert np.max(abs(change - explanation)) < 1e-7
    checks.append("Release-level news sums to every fixed-parameter update, and news plus reestimation reconciles consecutive predictions.")
    model_files = list((RESULTS / "models").glob("*.json"))
    assert len(model_files) == 54
    for path in model_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        train = training_window(panel, record["fit_asof"])
        for key in panel:
            np.testing.assert_allclose(record["standardization_mean"][key], train[key].mean(), atol=1e-10)
            np.testing.assert_allclose(record["standardization_std"][key], train[key].std(), atol=1e-10)
        history = np.array(record["llf_history"])
        assert np.isfinite(history).all()
        assert (np.diff(history) >= -1e-4).all(), path.name
        assert record["converged"], path.name
    checks.append("All 54 fits converged with nondecreasing likelihood; saved scaling exactly matches each release-masked rolling training vintage.")
    metrics = pd.read_csv(RESULTS / "summary_metrics.csv")
    for _, row in metrics.iterrows():
        rows = predictions.loc[(predictions.factors == row.factors) & (predictions.type == row.type)
                               & predictions.target.between("2021Q4", "2025Q3")]
        column = "prediction" if row.model.startswith("DFM") else {
            "Last released GDP": "last_gdp", "Historical mean": "historical_mean", "AR(1)": "ar1"}[row.model]
        error = rows[column] - rows.actual_log_yoy
        np.testing.assert_allclose(row.rmse, np.sqrt(np.mean(error ** 2)), rtol=1e-8)
    points = pd.read_csv(RESULTS / "paper_point_comparisons.csv")
    assert len(points) == 6 and points.reconstructed_prediction.notna().all()
    checks.append("Accuracy metrics recompute from the saved forecasts; all six published-point comparisons are present.")
    annual = pd.read_csv(RESULTS / "annual_forecasts.csv")
    assert np.isfinite(annual.prediction_pct).all()
    assert annual.loc[annual.year == 2025, "actual_pct"].isna().all()
    checks.append("Annual predictions are finite, and unavailable full-year 2025 actual GDP is not fabricated.")
    for document in [ROOT / "report.html", ROOT / "README.html"]:
        links = Links()
        links.feed(document.read_text(encoding="utf-8"))
        for relative in links.paths:
            target = document.parent.joinpath(*relative.replace("\\", "/").split("/"))
            assert target.is_file(), str(target)
    figures = list((RESULTS / "figures").glob("*.svg"))
    assert len(figures) == 7
    for figure in figures:
        ET.parse(figure)
    checks.append("Report and instructions have no broken local links; all seven SVG figures parse successfully.")
    summary = {"validated_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
               "behavior_tests": tests.testsRun, "checks": checks,
               "maximum_saved_news_discrepancy_pp": float(residual.abs().max())}
    write_json(RESULTS / "validation.json", summary)
    print("\n".join(checks))


if __name__ == "__main__":
    with output_session():
        validate()
