"""Generate static figures and an auditable local HTML results report."""
from io import BytesIO
from pathlib import Path
import html
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from file_utils import output_session, write_bytes, write_text

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
COLORS = ["#167d9a", "#dc782f", "#5261af", "#31976e", "#b95a83", "#899d3c", "#7e7470", "#399fab", "#b78d37", "#8247a7", "#4b8c66", "#444444"]


def save_figure(fig, name):
    stream = BytesIO()
    fig.savefig(stream, format="svg", bbox_inches="tight", metadata={"Date": None})
    write_bytes(FIGURES / f"{name}.svg", stream.getvalue())
    plt.close(fig)


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda value: f"{value:,.3f}", na_rep="—", escape=True)


def create_report():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=COLORS),
                         "figure.facecolor": "white", "axes.titleweight": "bold"})
    configuration = json.loads((RESULTS / "run_configuration.json").read_text(encoding="utf-8"))
    coverage = pd.read_csv(RESULTS / "data_coverage.csv")
    predictions = pd.read_csv(RESULTS / "quarterly_forecasts.csv", parse_dates=["asof"])
    common = pd.read_csv(RESULTS / "common_components.csv")
    panel = pd.read_csv(RESULTS / "transformed_panel.csv")
    panel["date"] = pd.PeriodIndex(panel.month, freq="M").to_timestamp()
    metrics = pd.read_csv(RESULTS / "summary_metrics.csv")
    rounds = pd.read_csv(RESULTS / "forecast_metrics.csv")
    annual = pd.read_csv(RESULTS / "annual_forecasts.csv", parse_dates=["asof"])
    diagnostics = pd.read_csv(RESULTS / "estimation_diagnostics.csv")
    source_path = ROOT / "data" / "sources.json"
    sources = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    main = predictions.loc[predictions.factors == 2]
    if main.empty:
        main = predictions.loc[predictions.factors == predictions.factors.min()]
    keys = [key for key in panel if key not in {"month", "date"}]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(panel[keys].notna().T, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ticks = [i for i, month in enumerate(panel.month) if month.endswith("-01") and int(month[:4]) % 2 == 0]
    ax.set_xticks(ticks, [panel.month.iloc[i][:4] for i in ticks])
    ax.set_yticks(range(len(keys)), keys)
    ax.set_title("Input availability after transformations · latest public vintage")
    save_figure(fig, "input_availability")
    fig, ax = plt.subplots(figsize=(12, 4))
    for count, group in common.groupby("factors"):
        dates = pd.PeriodIndex(group.month, freq="M").to_timestamp()
        ax.plot(dates, group.gdp_common_component, label=f"{count}-factor common component", alpha=0.85)
    actual = panel.loc[panel.gdp.notna()]
    ax.plot(actual.date, actual.gdp, "k.", label="GDP log y/y", markersize=3)
    ax.axvline(pd.Timestamp("2021-09-21"), color="#888", linestyle="--", linewidth=1)
    ax.set(ylabel="100 × log year-on-year change", title="GDP common components · initial coefficients, full-vintage smoother")
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, "common_components")
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharey=True)
    for ax, target in zip(axes.ravel(), ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]):
        rows = main.loc[main.target == target].sort_values("asof")
        if len(rows):
            ax.fill_between(rows["asof"].to_numpy(), rows.lower95.to_numpy(float), rows.upper95.to_numpy(float), color=COLORS[0], alpha=0.12)
            ax.plot(rows["asof"], rows.prediction, "o-", markersize=3, label=f"DFM{int(main.factors.iloc[0])}")
            if rows.actual_log_yoy.notna().any():
                ax.axhline(rows.actual_log_yoy.dropna().iloc[0], color="#444", linestyle="--", label="Latest actual")
        ax.set(title=target, ylabel="GDP log y/y (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", rotation=25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Quarterly nowcasts and backcasts · shaded bands are conditional 95% intervals", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "quarterly_nowcasts")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    scored = main.loc[main.target.between("2021Q4", "2025Q3") & main.error.notna()]
    axes[0].scatter(scored["round"], scored.error, s=12, alpha=0.45)
    average = scored.groupby("round").error.mean()
    axes[0].plot(average.index, average.values, "ko-", markersize=4, label="Mean error")
    axes[0].axhline(0, color="#aaa", linewidth=1)
    axes[0].axvline(6.5, color="#aaa", linestyle="--")
    axes[0].set(xlabel="Update round (1–6 nowcast; 7–12 backcast)", ylabel="Prediction − actual (p.p.)", title="Reconstructed forecast errors")
    axes[0].legend(frameon=False)
    selected = rounds.loc[rounds.factors == int(main.factors.iloc[0])]
    for name, rows in selected.groupby("model"):
        axes[1].plot(rows["round"], rows.rmse, "o-", markersize=3, label=name)
    axes[1].set(xlabel="Update round", ylabel="RMSE (p.p.)", title="Against transparent statistical benchmarks")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "forecast_errors")
    news_path = RESULTS / "news_details.csv"
    if news_path.stat().st_size > 5:
        news = pd.read_csv(news_path)
        if len(news):
            news = news.loc[news.factors == int(main.factors.iloc[0])].copy()
            news["absolute_impact"] = news.impact.abs()
            average = news.groupby(["asof", "target", "series"]).impact.sum().abs().groupby("series").sum()
            average /= main[["asof", "target"]].drop_duplicates().shape[0] - 2
            average = average.sort_values()
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.barh(average.index, average.values, color=COLORS[0])
            ax.set(xlabel="Mean absolute release impact per forecast update (p.p.)", title="News attribution · zero-impact dates included")
            save_figure(fig, "news_impacts")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, year in zip(axes.ravel(), [2022, 2023, 2024, 2025]):
        rows = annual.loc[(annual.factors == int(main.factors.iloc[0])) & (annual.year == year)]
        ax.plot(rows["asof"], rows.prediction_pct, label="DFM annual growth")
        if rows.actual_pct.notna().any():
            ax.axhline(rows.actual_pct.dropna().iloc[0], color="#444", linestyle="--", label="Latest actual")
        ax.set(title=str(year), ylabel="Annual real GDP growth (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", rotation=25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Calendar-year forecasts · aggregated quarterly levels, not mean quarterly growth", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "annual_nowcasts")
    nrows = (len(keys) - 1 + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, nrows * 2.5), squeeze=False)
    for ax, key in zip(axes.ravel(), [key for key in keys if key != "gdp"]):
        ax.plot(panel.date, panel[key], linewidth=1)
        ax.set_title(key)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color="#aaa", alpha=0.15)
    for ax in axes.ravel()[len(keys)-1:]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "indicator_series")
    summary = metrics.loc[(metrics.factors == int(main.factors.iloc[0]))][["type", "model", "n", "quarters", "rmse", "mae", "bias"]]
    sensitivity = metrics.loc[metrics.model.str.startswith("DFM"), ["model", "type", "n", "quarters", "rmse", "mae", "bias"]]
    best_variants = sensitivity.loc[sensitivity.groupby("type").rmse.idxmin()]
    sensitivity_note = "; ".join(f"{row.type}: {row.model}" for _, row in best_variants.iterrows())
    validation_path = RESULTS / "validation.json"
    validation_note = ""
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("passed"):
            validation_note = f'<p>Saved validation: {validation["behavior_tests"]} behavior tests passed, plus empirical timing, checksum, news and report-asset checks. <a href="results/validation.json">Validation details</a> · <a href="results/test_results.txt">Test log</a>.</p>'
    latest = main.loc[main.target == "2025Q4"].sort_values("asof").tail(1)
    last_text = f"{latest.prediction.iloc[0]:.2f}% log y/y on {latest['asof'].iloc[0].date()}" if len(latest) else "Not estimated"
    omitted = ", ".join(configuration["omitted_series"]) or "None"
    first_date = configuration["created_utc"][:10]
    primary_metrics = metrics.loc[metrics.factors == int(main.factors.iloc[0])]
    def rmse(kind, model):
        return float(primary_metrics.loc[(primary_metrics.type == kind) & (primary_metrics.model == model), "rmse"].iloc[0])
    primary_name = f"DFM{int(main.factors.iloc[0])}"
    findings = (f"The reconstructed {primary_name} nowcast RMSE is {rmse('nowcast', primary_name):.3f} p.p., "
                f"versus {rmse('nowcast', 'AR(1)'):.3f} for AR(1); its backcast RMSE is "
                f"{rmse('backcast', primary_name):.3f}, versus {rmse('backcast', 'AR(1)'):.3f} for AR(1). "
                "Lower RMSE is better. These public-data results do not reproduce the paper’s favorable comparative performance; "
                "they do not establish that the published results are incorrect because the source data and vintages differ.")
    point_path = RESULTS / "paper_point_comparisons.csv"
    point_table = table(pd.read_csv(point_path)) if point_path.exists() and point_path.stat().st_size > 10 else "Not generated in this run."
    figures = [("input_availability", "1. Data coverage"), ("common_components", "2. Common GDP components"),
               ("quarterly_nowcasts", "3. Quarterly predictions"), ("forecast_errors", "4. Forecast evaluation"),
               ("news_impacts", "5. News attribution"), ("annual_nowcasts", "6. Calendar-year growth"),
               ("indicator_series", "7. Monthly inputs")]
    gallery = "".join(f'<section><h2>{title}</h2><a href="results/figures/{name}.svg"><img src="results/figures/{name}.svg" alt="{title}"></a></section>'
                      for name, title in figures if (FIGURES / f"{name}.svg").exists())
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Danov et al. · Public-data reconstruction</title><style>
body{{margin:0;background:#f3f6f8;color:#213442;font:16px/1.6 system-ui,Segoe UI,sans-serif}}main{{max-width:1160px;margin:auto;padding:42px 24px}}h1{{font-size:36px;line-height:1.15;margin-bottom:12px}}h2{{font-size:23px}}a{{color:#126c8e}}section,.card{{background:white;padding:24px 28px;margin:24px 0;border:1px solid #dce4e8;border-radius:12px}}.badge{{color:#915200;background:#fff1cf;display:inline-block;padding:5px 12px;border-radius:6px;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}.grid .card{{margin:0}}.number{{font-size:23px;font-weight:700;color:#146982}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #e0e7eb;white-space:nowrap}}th{{background:#eff4f6}}code,pre{{background:#eef3f6;padding:3px 6px}}pre{{overflow:auto;padding:16px;font-size:13px}}small{{color:#526975}}li{{margin-bottom:8px}}details{{margin:16px 0}}footer{{font-size:13px;color:#627683}}</style></head><body><main>
<div class="badge">PUBLIC-DATA RECONSTRUCTION · NOT AN EXACT REPLICATION</div>
<h1>Nowcasting GDP Growth for Kenya</h1><p>Reconstruction of Danov, Giannone, Kabundi, Okou &amp; Spilimbergo · IMF Working Paper 26/32 (2026)</p>
<p><a href="Danov_et_al_2026_Nowcasting_GDP_Growth_for_Kenya.pdf">Original paper (68 pages)</a> · <a href="README.html">Run instructions &amp; methodological notes</a> · Generated {first_date}</p>
<div class="grid"><div class="card"><small>Public input series</small><div class="number">{len(configuration['included_series'])} of 12</div><small>Omitted: {html.escape(omitted)}</small></div>
<div class="card"><small>Forecast exercise</small><div class="number">{configuration['forecast_dates']} dates</div><small>September 2021–December 2025 · factor counts: {', '.join(map(str, configuration['factors']))}</small></div>
<div class="card"><small>Latest 2025 Q4 nowcast</small><div class="number">{last_text}</div><small>Reconstructed input vintage; not the authors’ estimate</small></div></div>
<section><h2>How to interpret these results</h2><p>This is an executed reconstruction using real public observations. No unavailable indicator has been replaced with synthetic data, annual GDP has not been interpolated into quarters, and the paper’s reported numerical results are not used to fit the model. Differences in data, revisions and measurement mean that this is a test of a reconstructed method, not confirmation of the paper’s exact figures.</p>
<ul><li>Year-on-year transformations: 100 log differences, with quarterly GDP on quarter-end months. PMI and visitor arrivals remain in levels. Nonpositive levels are treated as missing.</li><li>One, two and three joint VAR(1) factors, individual AR(1) components, EM estimation, and Kalman filtering/smoothing. GDP uses the paper’s [1,1,1] aggregation; the standard quarter-on-quarter [1,2,3,2,1] rule is not substituted.</li><li>Only lag-eligible observations enter each vintage. Parameters and standardization are fitted initially and at quarter starts using up to twelve years, then frozen within that quarter.</li><li>All forecast dates are pseudo-real-time, including April 2024 onward. Historical revisions and the Haver release calendar are unavailable. Fixed Table 1 end-of-period day lags replace exact release dates.</li><li>Conditional intervals exclude parameter uncertainty, vintage revisions and structural breaks. Full-vintage smoothed components are descriptive and never used as forecast inputs.</li><li>The evaluation contains {configuration['evaluation_quarters']} quarters over 2021Q4–2025Q3. The paper calls this span “15 quarters,” but the inclusive calendar span is 16; actual counts are reported rather than silently dropping a quarter.</li><li>CBK and WEO forecast-vintage comparisons are not reconstructed without their original dated forecast series. Statistical baselines are clearly labeled alternatives.</li></ul></section>
<section><h2>Principal finding</h2><p>{findings}</p><p>The requested factor counts were retained as planned; none was selected after seeing the evaluation errors. The saved CSV includes each model’s results.</p></section>
<section><h2>Forecast accuracy · reconstructed {primary_name}</h2><p>Percentage points of 100 × log year-on-year GDP growth. Every model is evaluated on the same dated targets. Errors use the acquired latest GDP vintage, not first releases. Nowcasts and backcasts each combine six update rounds per completed target quarter.</p><div class="scroll">{table(summary)}</div><p>Do not interpret deviations from the authors’ estimates as implementation error alone, or favorable benchmark results as an exact replication.</p></section>
<section><h2>Factor-count robustness</h2><p>Lowest reconstructed RMSE among the requested DFM variants: {sensitivity_note}. This is a reported sensitivity result, not an ex-post change to the main specification.</p><div class="scroll">{table(sensitivity)}</div></section>
{gallery}
<section><h2>Published point estimates versus this reconstruction</h2><p>Selected dated values explicitly described in Section 4.2 (PDF page 30). Differences are disclosed, not minimized by calibration. The paper’s percentage-growth descriptions and the implemented log-growth definition are not necessarily numerically identical.</p><div class="scroll">{point_table}</div></section>
<section><h2>Input coverage</h2><p>Arrivals use a JKIA+MIA airport-only proxy. Three July–September 2009 electricity observations are excluded because of inconsistent source date labels, and October 2010 airport arrivals are missing. No values are interpolated. The three electricity gaps also remove their 2010 year-on-year comparisons. <a href="data/source_quality_issues.json">Source-quality record</a>.</p><div class="scroll">{table(coverage)}</div></section>
<section><h2>Estimation &amp; reconciliation</h2>{validation_note}<p>{configuration['converged_fits']} of {configuration['total_fits']} quarterly EM fits reached the configured tolerance. Nonconverged fits, if any, remain flagged and are not represented as converged. Maximum absolute discrepancy between release-news contributions and the fixed-parameter prediction change: {configuration['maximum_news_reconciliation_error']:.3g} p.p.</p>
<details><summary>Quarterly fit diagnostics</summary><div class="scroll">{table(diagnostics[["factors", "fit_asof", "iterations", "loglikelihood", "converged"]])}</div></details></section>
<section><h2>Downloads &amp; provenance</h2><ul><li><a href="results/quarterly_forecasts.csv">Every quarterly forecast and decomposition</a></li><li><a href="results/annual_forecasts.csv">Every calendar-year forecast</a></li><li><a href="results/news_details.csv">Release-level surprises, weights and impacts</a></li><li><a href="results/summary_metrics.csv">Summary accuracy measures</a> · <a href="results/forecast_metrics.csv">Metrics by update round</a></li><li><a href="results/table1_reconstruction.csv">Table 1 descriptive comparisons</a></li><li><a href="results/data_coverage.csv">Data coverage</a> · <a href="results/transformed_panel.csv">Transformed panel</a> · <a href="data/observations.csv">Normalized raw observations</a></li><li><a href="data/sources.json">Source metadata and acquisition dates</a> · <a href="file_inventory.json">File checksums</a></li><li><a href="results/estimation_diagnostics.csv">Estimation diagnostics</a> · <a href="results/run_configuration.json">Run configuration</a> · <a href="results/environment.json">Software versions</a></li></ul>
<details><summary>Source metadata</summary><pre>{html.escape(json.dumps(sources, indent=2, ensure_ascii=False))}</pre></details></section>
<footer>Source paper: https://www.imf.org/-/media/files/publications/wp/2026/english/wpiea2026032-source-pdf.pdf · DOI and bibliographic details are in the saved paper. This report contains independent analysis; no affiliation with the authors or IMF is implied.</footer></main></body></html>"""
    write_text(ROOT / "report.html", body)
    print(f"Report written: {ROOT / 'report.html'}", flush=True)


if __name__ == "__main__":
    with output_session():
        create_report()
