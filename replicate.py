"""Run the public-data reconstruction. See README.html for sources and limitations."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from data_model import SERIES, baseline_forecasts, forecast_dates, training_window, transform_panel, vintage
from factor_model import make_model
from file_utils import output_session, write_json, write_text

RESULTS = ROOT / "results"


def save_csv(frame, name):
    write_text(RESULTS / name, frame.to_csv(index=False, float_format="%.10g"))


def signal(result, quarter):
    month = pd.Period(quarter, freq="Q").asfreq("M", how="end")
    t = result.model._index.get_loc(month)
    z = result.model["design"][-1]
    mean = float(z @ result.smoothed_state[:, t])
    variance = float(z @ result.smoothed_state_cov[:, :, t] @ z)
    scale = float(result.model._endog_std.iloc[-1])
    location = float(result.model._endog_mean.iloc[-1])
    return mean * scale + location, np.sqrt(max(variance, 0)) * scale


def apply_vintage(fit, panel, asof, end):
    data = vintage(panel, asof).loc[fit.model._index[0]:end]
    return fit.apply(data, retain_standardization=True, copy_initialization=True)


def model_record(fit, asof, factor_count, caught, tolerance):
    history = np.asarray(fit.mle_retvals.llf, dtype=float)
    delta = 2 * abs(history[-1] - history[-2]) / (abs(history[-1]) + abs(history[-2])) if len(history) > 1 else None
    return {
        "factors": factor_count, "fit_asof": str(pd.Timestamp(asof).date()),
        "train_start": str(fit.model._index[0]), "train_end": str(fit.model._index[-1]),
        "observations": int(fit.nobs), "iterations": int(fit.mle_retvals.iter),
        "loglikelihood": float(fit.llf), "last_relative_llf_change": delta,
        "converged": bool(delta is not None and delta <= tolerance),
        "warnings": sorted(set(str(w.message) for w in caught)),
        "parameters": dict(zip(fit.param_names, map(float, fit.params))),
        "standardization_mean": fit.model._endog_mean.to_dict(),
        "standardization_std": fit.model._endog_std.to_dict(),
        "initial_state": fit.filter_results.initial_state.tolist(),
        "initial_state_covariance": fit.filter_results.initial_state_cov.tolist(),
        "llf_history": history.tolist(),
    }


def annual_growth(result, levels, asof, year):
    known = levels.loc[levels.index.to_timestamp(how="end").normalize() + pd.Timedelta(days=SERIES["gdp"]["lag"]) <= asof]
    estimates = {quarter: float(value) for quarter, value in known.items()}
    first = min(estimates) if estimates else None
    if first is None:
        return np.nan
    for quarter in pd.period_range(first, f"{year}Q4", freq="Q"):
        if quarter not in estimates:
            previous = estimates.get(quarter - 4)
            if previous is None:
                continue
            try:
                predicted, _ = signal(result, quarter)
            except KeyError:
                continue
            estimates[quarter] = previous * np.exp(predicted / 100)
    current = [estimates.get(pd.Period(f"{year}Q{q}", freq="Q"), np.nan) for q in range(1, 5)]
    previous = [estimates.get(pd.Period(f"{year-1}Q{q}", freq="Q"), np.nan) for q in range(1, 5)]
    return 100 * (sum(current) / sum(previous) - 1)


def run(args):
    RESULTS.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "observations.csv")
    panel, coverage = transform_panel(raw)
    save_csv(panel.rename_axis("month").reset_index(), "transformed_panel.csv")
    save_csv(coverage, "data_coverage.csv")
    raw_gdp = raw.loc[raw.series == "gdp"]
    levels = pd.Series(raw_gdp.value.to_numpy(float), index=pd.PeriodIndex(raw_gdp.period, freq="Q")).sort_index().loc[:"2025Q3"]
    dates = forecast_dates()
    forecasts, impacts, annual, fits, components, descriptive = [], [], [], [], [], []
    published_correlations = {"pmi": -0.01, "remittances": -0.02, "mobile_money": 0.02,
                              "exports": 0.19, "imports": 0.36, "coffee": -0.02, "tea": -0.18,
                              "m2": 0.11, "electricity": 0.30, "arrivals": 0.38, "vehicles": 0.35, "gdp": 1.0}
    for factor_count in args.factors:
        fit, previous_fit, previous_date, regime = None, None, None, None
        for number, asof in enumerate(dates):
            quarter = asof.to_period("Q")
            if regime != quarter:
                fit_asof = asof if fit is None else quarter.start_time.normalize()
                train = training_window(panel, fit_asof)
                if train.gdp.notna().sum() < 20:
                    raise ValueError("At least twenty quarterly GDP observations are required for estimation.")
                model = make_model(train, factor_count)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = model.fit_em(maxiter=args.maxiter, tolerance=args.tolerance, disp=False)
                record = model_record(fit, fit_asof, factor_count, caught, args.tolerance)
                write_json(RESULTS / "models" / f"dfm{factor_count}_{str(fit_asof.date())}.json", record)
                fits.append({k: v for k, v in record.items() if k not in {
                    "parameters", "initial_state", "initial_state_covariance", "llf_history", "standardization_mean", "standardization_std"}})
                print(f"DFM{factor_count} fit {str(fit_asof.date())}: {record['iterations']} iterations; converged={record['converged']}", flush=True)
                if regime is None:
                    full = fit.apply(panel.loc[train.index[0]:], retain_standardization=True, copy_initialization=True)
                    z = full.model["design"]
                    kstates = full.model._s.k_states_factors
                    common = z[:, :kstates] @ full.smoothed_state[:kstates]
                    common = common.T * full.model._endog_std.to_numpy() + full.model._endog_mean.to_numpy()
                    for i, month in enumerate(full.model._index):
                        components.append({"month": str(month), "factors": factor_count,
                                           "gdp_common_component": float(common[i, -1])})
                    if factor_count == 2:
                        train_common = (fit.model["design"][:, :kstates] @ fit.smoothed_state[:kstates]).T
                        train_common *= fit.model._endog_std.to_numpy()
                        for i, key in enumerate(panel.columns):
                            quarterly = panel[key].groupby(panel.index.asfreq("Q")).mean()
                            actual = panel.gdp.dropna().copy()
                            actual.index = actual.index.asfreq("Q")
                            paired = pd.concat([quarterly.rename("x"), actual.rename("y")], axis=1).loc["2010Q1":"2021Q3"].dropna()
                            correlation, pvalue = pearsonr(paired.x, paired.y) if len(paired) > 3 else (np.nan, np.nan)
                            valid = train[key].notna().to_numpy()
                            share = float(np.var(train_common[valid, i]) / np.var(train[key].dropna()))
                            descriptive.append({"series": key, "name": SERIES[key]["name"],
                                                "paper_correlation": published_correlations[key],
                                                "reconstructed_correlation": correlation, "pvalue": pvalue,
                                                "n_quarters": len(paired), "common_variance_ratio": share})
                regime = quarter
            horizon_end = pd.Period(f"{asof.year}-12", freq="M")
            current = apply_vintage(fit, panel, asof, horizon_end)
            previous_same_model = apply_vintage(fit, panel, previous_date, horizon_end) if previous_date is not None else None
            previous_old_model = apply_vintage(previous_fit, panel, previous_date, horizon_end) if previous_fit is not None else None
            available = vintage(panel, asof)
            for target in (quarter, quarter - 1):
                month = target.asfreq("M", how="end")
                point, sigma = signal(current, target)
                actual = panel.loc[month, "gdp"]
                release_effect = parameter_effect = residual = 0.0
                news_details = None
                if previous_same_model is not None:
                    old_current, _ = signal(previous_same_model, target)
                    old_previous, _ = signal(previous_old_model, target)
                    release_effect = point - old_current
                    parameter_effect = old_current - old_previous
                    if args.news:
                        news = current.news(previous_same_model, impact_date=month,
                                            impacted_variable="gdp", comparison_type="previous")
                        news_details = news.details_by_impact.reset_index()
                        explained = float(news_details.impact.sum()) if len(news_details) else 0.0
                        residual = release_effect - explained
                        if abs(residual) > 1e-6:
                            raise AssertionError(f"News reconciliation failed for {asof}, {target}: {residual}")
                        for _, row in news_details.iterrows():
                            impacts.append({"asof": str(asof.date()), "target": str(target), "factors": factor_count,
                                            "series": row["updated variable"], "observation_month": str(row["update date"]),
                                            "observed": row.observed, "expected": row["forecast (prev)"],
                                            "news": row.news, "weight": row.weight, "impact": row.impact})
                offset = (asof.year - target.year) * 12 + asof.month - (target.quarter - 1) * 3 - 1
                round_number = offset * 2 + (1 if asof.day == 7 else 2)
                forecasts.append({"asof": str(asof.date()), "target": str(target), "factors": factor_count,
                                  "type": "nowcast" if target == quarter else "backcast", "round": round_number,
                                  "prediction": point, "conditional_sd": sigma, "lower95": point - 1.96 * sigma,
                                  "upper95": point + 1.96 * sigma, "actual_log_yoy": actual,
                                  "error": point - actual if np.isfinite(actual) else np.nan,
                                  "release_news_effect": release_effect, "reestimation_effect": parameter_effect,
                                  "news_reconciliation_error": residual,
                                  **baseline_forecasts(available, target)})
            for year in (asof.year - 1, asof.year):
                if 2022 <= year <= 2025:
                    value = annual_growth(current, levels, asof, year)
                    actual_levels = levels.loc[str(year)]
                    prior_levels = levels.loc[str(year - 1)]
                    actual_annual = 100 * (actual_levels.sum() / prior_levels.sum() - 1) if len(actual_levels) == len(prior_levels) == 4 else np.nan
                    annual.append({"asof": str(asof.date()), "year": year, "factors": factor_count,
                                   "prediction_pct": value, "actual_pct": actual_annual})
            previous_date, previous_fit = asof, fit
        print(f"Completed DFM{factor_count}: {len(dates)} forecast dates.", flush=True)
    predictions = pd.DataFrame(forecasts)
    save_csv(predictions, "quarterly_forecasts.csv")
    save_csv(pd.DataFrame(impacts), "news_details.csv")
    save_csv(pd.DataFrame(annual), "annual_forecasts.csv")
    save_csv(pd.DataFrame(components), "common_components.csv")
    save_csv(pd.DataFrame(descriptive), "table1_reconstruction.csv")
    save_csv(pd.DataFrame(fits).assign(warnings=lambda x: x.warnings.map(json.dumps)), "estimation_diagnostics.csv")
    evaluation = predictions.loc[predictions.target.between("2021Q4", "2025Q3") & predictions.error.notna()]
    metric_rows = []
    for (factor_count, kind, round_number), rows in evaluation.groupby(["factors", "type", "round"]):
        for label, column in [(f"DFM{factor_count}", "prediction"), ("Last released GDP", "last_gdp"),
                              ("Historical mean", "historical_mean"), ("AR(1)", "ar1")]:
            errors = rows[column] - rows.actual_log_yoy
            metric_rows.append({"factors": factor_count, "model": label, "type": kind, "round": int(round_number),
                                "n": int(errors.notna().sum()), "n_quarters": int(rows.target.nunique()),
                                "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(abs(errors))),
                                "bias": float(errors.mean())})
    save_csv(pd.DataFrame(metric_rows), "forecast_metrics.csv")
    summary_rows = []
    for factor_count, group in evaluation.groupby("factors"):
        for kind, rows in group.groupby("type"):
            for label, column in [(f"DFM{factor_count}", "prediction"), ("Last released GDP", "last_gdp"),
                                  ("Historical mean", "historical_mean"), ("AR(1)", "ar1")]:
                errors = rows[column] - rows.actual_log_yoy
                summary_rows.append({"factors": factor_count, "type": kind, "model": label,
                                     "n": int(errors.notna().sum()), "quarters": int(rows.target.nunique()),
                                     "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(abs(errors))),
                                     "bias": float(errors.mean())})
    save_csv(pd.DataFrame(summary_rows), "summary_metrics.csv")
    paper_points = [
        ("2024-01-07", "2024Q1", 5.06), ("2024-03-21", "2024Q1", 6.40),
        ("2024-04-07", "2024Q2", 5.78), ("2024-06-21", "2024Q2", 4.83),
        ("2025-01-07", "2025Q1", 4.16), ("2025-03-21", "2025Q1", 3.12),
    ]
    comparisons = []
    for origin, target, published in paper_points:
        match = predictions.loc[(predictions.factors == 2) & (predictions["asof"] == origin) & (predictions.target == target)]
        if len(match):
            reconstructed = float(match.prediction.iloc[0])
            comparisons.append({"asof": origin, "target": target, "paper_prediction": published,
                                "reconstructed_prediction": reconstructed, "difference_pp": reconstructed - published,
                                "paper_reference": "Section 4.2, printed page 28 (PDF page 30)",
                                "interpretation": "Illustrative comparison, not a numerical pass/fail: source vintages and definitions differ."})
    save_csv(pd.DataFrame(comparisons), "paper_point_comparisons.csv")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pymupdf", "pypdf", "lxml", "openpyxl", "requests"]}}
    write_json(RESULTS / "environment.json", environment)
    write_json(RESULTS / "run_configuration.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "factors": args.factors,
        "maxiter": args.maxiter, "tolerance": args.tolerance, "news": args.news,
        "start": "2021-09-21", "end": "2025-12-21", "raw_observations": len(raw),
        "included_series": list(panel.columns), "omitted_series": [s for s in SERIES if s not in panel],
        "forecast_dates": len(dates), "quarterly_forecast_rows": len(predictions),
        "converged_fits": int(sum(row["converged"] for row in fits)), "total_fits": len(fits),
        "maximum_news_reconciliation_error": float(predictions.news_reconciliation_error.abs().max()),
        "evaluation_quarters": int(evaluation.target.nunique()), "data_vintage": "Latest public data, truncated to Table 1 endpoints; not historical revisions.",
    })
    from reporting import create_report
    create_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--no-news", dest="news", action="store_false")
    options = parser.parse_args()
    if not set(options.factors) <= {1, 2, 3}:
        parser.error("Supported factor counts: 1, 2, 3")
    with output_session():
        run(options)
