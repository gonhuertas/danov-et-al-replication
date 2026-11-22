"""Focused checks for aggregation, missing data, news, and vintage isolation."""
import unittest
import warnings
import numpy as np
import pandas as pd

from data_model import baseline_forecasts, forecast_dates, training_window, transform_panel, vintage
from factor_model import make_model


def synthetic_panel():
    random = np.random.default_rng(1827)
    n = 132
    factors = np.zeros((n, 2))
    for t in range(1, n):
        factors[t] = np.array([[0.65, 0.1], [0.05, 0.45]]) @ factors[t - 1] + random.normal(size=2)
    loading = np.array([[1, 0.2], [0.5, 1], [-0.3, 0.8], [1.2, -0.4]])
    monthly = factors @ loading.T + random.normal(scale=0.4, size=(n, 4))
    frame = pd.DataFrame(monthly, index=pd.period_range("2000-01", periods=n, freq="M"),
                         columns=["remittances", "exports", "imports", "electricity"])
    latent = factors @ np.array([0.4, 0.6]) + random.normal(scale=0.3, size=n)
    gdp = pd.Series(latent).rolling(3).sum().to_numpy() + 5
    frame["gdp"] = np.where(frame.index.month % 3 == 0, gdp, np.nan)
    return frame


class DataChecks(unittest.TestCase):
    def test_pdf_year_suffix_does_not_consume_small_arrivals_counts(self):
        from download_data import parse_row_year
        self.assertEqual(parse_row_year("94 0 94", 2020, 2021), (2020, "94 0 94"))
        self.assertEqual(parse_row_year("12  83,122 16,135", 2012, 2012), (2012, "83,122 16,135"))
        self.assertEqual(parse_row_year("-10 186.69 454.60", 2009, 2009), (2010, "186.69 454.60"))

    def test_transform_is_log_yoy_and_does_not_fill_gaps(self):
        months = pd.period_range("2008-01", "2011-12", freq="M")
        raw = pd.DataFrame({"series": "exports", "period": months.astype(str),
                            "value": np.exp(np.arange(len(months)) * 0.01)})
        quarters = pd.period_range("2008Q1", "2011Q4", freq="Q")
        gdp = pd.DataFrame({"series": "gdp", "period": quarters.astype(str),
                            "value": np.exp(np.arange(len(quarters)) * 0.02)})
        raw.loc[raw.period == "2010-03", "value"] = np.nan
        panel, _ = transform_panel(pd.concat([raw, gdp]), end="2011-12")
        self.assertAlmostEqual(panel.loc["2010-02", "exports"], 12)
        self.assertTrue(np.isnan(panel.loc["2010-03", "exports"]))
        self.assertAlmostEqual(panel.loc["2010-03", "gdp"], 8)
        self.assertEqual(panel.gdp.notna().sum(), 8)
        self.assertTrue(panel.loc[panel.index.month % 3 != 0, "gdp"].isna().all())

    def test_release_boundary_and_future_mutation(self):
        panel = synthetic_panel()
        old = vintage(panel, "2010-06-10")
        self.assertTrue(np.isnan(old.loc["2010-05", "remittances"]))
        released = vintage(panel, "2010-06-11")
        self.assertEqual(released.loc["2010-05", "remittances"], panel.loc["2010-05", "remittances"])
        changed = panel.copy()
        for key in changed:
            hidden = old[key].isna() & panel[key].notna()
            changed.loc[hidden, key] = 1e9
        pd.testing.assert_frame_equal(training_window(panel, "2010-06-10"),
                                      training_window(changed, "2010-06-10"))
        self.assertEqual(baseline_forecasts(old, "2010Q2"),
                         baseline_forecasts(vintage(changed, "2010-06-10"), "2010Q2"))

    def test_annual_growth_uses_quarterly_level_weights_and_hides_unreleased_actuals(self):
        from unittest.mock import patch
        from replicate import annual_growth
        quarters = pd.period_range("2020Q1", "2022Q4", freq="Q")
        levels = pd.Series([100, 200, 300, 400, 110, 210, 310, 410, 9999, 9999, 9999, 9999], index=quarters)
        rates = {1: 0.10, 2: 0.20, 3: 0.0, 4: -0.10}
        expected = 100 * ((110 * 1.1 + 210 * 1.2 + 310 + 410 * 0.9) / 1040 - 1)
        with patch("replicate.signal", side_effect=lambda result, q: (100 * np.log1p(rates[q.quarter]), 0)):
            actual = annual_growth(None, levels, pd.Timestamp("2022-05-21"), 2022)
        self.assertAlmostEqual(actual, expected)

    def test_schedule_starts_at_paper_origin(self):
        dates = forecast_dates()
        self.assertEqual(str(dates[0].date()), "2021-09-21")
        self.assertEqual(str(dates[-1].date()), "2025-12-21")
        self.assertTrue(all(date.day in (7, 21) for date in dates))


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_panel()
        cls.old = cls.data.copy()
        cls.old.iloc[-6:, :] = np.nan
        cls.model = make_model(cls.old, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.fit = cls.model.fit_em(maxiter=45, tolerance=1e-5, disp=False)

    def test_quarterly_design_sums_three_months_including_idiosyncratic(self):
        model = self.model
        model.update(self.fit.params)
        factors = model._s["factors_L1_5_ix"]
        row = model["design"][-1]
        for lag in (1, 2):
            np.testing.assert_allclose(row[factors[lag]], row[factors[0]])
        np.testing.assert_allclose(row[factors[3:]], 0)
        np.testing.assert_allclose(row[model._s["idio_ar_Q_ix"][0]], [1, 1, 1, 0, 0])
        self.assertTrue(np.isfinite(self.fit.llf))
        self.assertTrue(np.isfinite(self.fit.get_prediction(end=len(self.data) - 1).predicted_mean.gdp).all())
        likelihood = np.asarray(self.fit.mle_retvals.llf)
        self.assertTrue((np.diff(likelihood) >= -0.001).all())

    def test_news_reconciles_with_forecast_update_and_cloning(self):
        new = self.old.copy()
        new.iloc[-4, 0] = self.data.iloc[-4, 0]
        updated = self.fit.apply(new, retain_standardization=True, copy_initialization=True)
        self.assertEqual(type(updated.model), type(self.model))
        news = updated.news(self.fit, impact_date=self.data.index[-1], impacted_variable="gdp",
                            comparison_type="previous")
        before = self.fit.get_prediction().predicted_mean.gdp.iloc[-1]
        after = updated.get_prediction().predicted_mean.gdp.iloc[-1]
        self.assertAlmostEqual(float(news.total_impacts.gdp.iloc[0]), after - before, places=6)
        self.assertGreater(abs(after - before), 1e-8)
        self.assertAlmostEqual(float(news.details_by_impact.impact.sum()), after - before, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
