"""Paper input definitions, transformations, and pseudo-real-time availability."""
import numpy as np
import pandas as pd

SERIES = {
    "pmi": {"name": "Stanbic Bank Kenya PMI", "frequency": "M", "lag": 5, "transform": "level", "end": "2025-12", "unit": "Index"},
    "remittances": {"name": "Diaspora remittances", "frequency": "M", "lag": 11, "transform": "log12", "end": "2025-11", "unit": "Million USD"},
    "mobile_money": {"name": "Total agent cash in cash out", "frequency": "M", "lag": 17, "transform": "log12", "end": "2025-11", "unit": "Billion KES"},
    "exports": {"name": "Exports of goods", "frequency": "M", "lag": 41, "transform": "log12", "end": "2025-11", "unit": "Million KES"},
    "imports": {"name": "Imports of goods", "frequency": "M", "lag": 41, "transform": "log12", "end": "2025-11", "unit": "Million KES"},
    "coffee": {"name": "Coffee exports", "frequency": "M", "lag": 41, "transform": "log12", "end": "2025-11", "unit": "Tonnes"},
    "tea": {"name": "Tea exports", "frequency": "M", "lag": 41, "transform": "log12", "end": "2025-11", "unit": "Tonnes"},
    "m2": {"name": "Money supply M2", "frequency": "M", "lag": 45, "transform": "log12", "end": "2025-09", "unit": "Million KES"},
    "electricity": {"name": "Electricity production", "frequency": "M", "lag": 48, "transform": "log12", "end": "2025-11", "unit": "Million kWh"},
    "arrivals": {"name": "Visitor arrivals", "frequency": "M", "lag": 48, "transform": "level", "end": "2025-09", "unit": "Persons"},
    "vehicles": {"name": "Vehicle production", "frequency": "M", "lag": 48, "transform": "log12", "end": "2025-11", "unit": "Units"},
    "gdp": {"name": "Real GDP", "frequency": "Q", "lag": 99, "transform": "log4", "end": "2025Q3", "unit": "Million KES, constant prices"},
}


def transform_panel(raw, start="2010-01", end="2025-12"):
    index = pd.period_range(start, end, freq="M")
    panel = pd.DataFrame(index=index)
    diagnostics = []
    for key, spec in SERIES.items():
        rows = raw.loc[raw.series == key].copy()
        if rows.empty:
            diagnostics.append({"series": key, "status": "omitted", "reason": "No usable public history acquired"})
            continue
        frequency = "Q-DEC" if spec["frequency"] == "Q" else "M"
        periods = pd.PeriodIndex(rows.period, freq=frequency)
        if periods.duplicated().any():
            raise ValueError(f"Duplicate observations for {key}")
        levels = pd.Series(rows.value.to_numpy(float), index=periods).sort_index()
        levels = levels.loc[:pd.Period(spec["end"], freq=frequency)]
        levels = levels.reindex(pd.period_range(levels.index.min(), levels.index.max(), freq=frequency))
        invalid = int(((levels <= 0) & levels.notna()).sum()) if spec["transform"] != "level" else 0
        if spec["transform"] == "level":
            transformed = levels.copy()
        else:
            positive = levels.where(levels > 0)
            lag = 4 if frequency == "Q-DEC" else 12
            transformed = 100 * (np.log(positive) - np.log(positive.shift(lag)))
        if frequency == "Q-DEC":
            transformed.index = transformed.index.asfreq("M", how="end")
        panel[key] = transformed.reindex(index)
        diagnostics.append({"series": key, "status": "included", "raw_start": str(levels.first_valid_index()),
                            "raw_end": str(levels.last_valid_index()), "n_raw": int(levels.notna().sum()),
                            "nonpositive_levels": invalid, "n_transformed": int(panel[key].notna().sum()),
                            "first_transformed": str(panel[key].first_valid_index()),
                            "last_transformed": str(panel[key].last_valid_index())})
    if "gdp" not in panel:
        raise ValueError("Actual quarterly GDP levels are required; annual interpolation is not supported.")
    columns = [key for key in SERIES if key in panel and key != "gdp"] + ["gdp"]
    return panel[columns], pd.DataFrame(diagnostics)


def release_dates(index, key):
    return index.to_timestamp(how="end").normalize() + pd.Timedelta(days=SERIES[key]["lag"])


def vintage(panel, asof):
    asof = pd.Timestamp(asof).normalize()
    result = panel.copy()
    for key in result.columns:
        result.loc[release_dates(result.index, key) > asof, key] = np.nan
    return result


def forecast_dates(start="2021-09-21", end="2025-12-21"):
    dates = []
    for month in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        for day in (7, 21):
            date = month.to_timestamp() + pd.Timedelta(days=day - 1)
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                dates.append(date)
    return dates


def training_window(panel, asof, years=12):
    available = vintage(panel, asof)
    last_month = pd.Timestamp(asof).to_period("M") - 1
    first_month = max(panel.index.min(), last_month - (years * 12 - 1))
    return available.loc[first_month:last_month]


def baseline_forecasts(available, target):
    observed = available.gdp.dropna()
    if observed.empty:
        return {"last_gdp": np.nan, "historical_mean": np.nan, "ar1": np.nan}
    dates = observed.index.asfreq("Q")
    target = pd.Period(target, freq="Q")
    steps = max(target.ordinal - dates[-1].ordinal, 0)
    frame = pd.Series(observed.to_numpy(), index=dates)
    frame = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq="Q"))
    pairs = pd.concat([frame.shift(1), frame], axis=1).dropna()
    if len(pairs) > 5:
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(pairs)), pairs.iloc[:, 0]]), pairs.iloc[:, 1], rcond=None)[0]
        slope = np.clip(slope, -0.99, 0.99)
        pred = observed.iloc[-1]
        for _ in range(steps):
            pred = intercept + slope * pred
    else:
        pred = observed.iloc[-1]
    return {"last_gdp": float(observed.iloc[-1]), "historical_mean": float(observed.mean()), "ar1": float(pred)}
