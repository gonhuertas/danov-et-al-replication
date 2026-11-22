# Nowcasting GDP Growth for Kenya: public-data reconstruction

An independent reconstruction of Danov, Giannone, Kabundi, Okou and Spilimbergo (2026), *Nowcasting GDP Growth for Kenya*, IMF Working Paper WP/26/32.

**This is not an exact numerical replication.** The original authors' code and data vintages were not available. Full historical PMI is omitted, arrivals use a consistent JKIA + MIA airport proxy, and the GDP input combines two documented official vintages. All departures and unresolved source gaps are recorded.

## Contents

- `report.html`: main results report, with seven vector figures and links to results.
- `README.html`: detailed methods, limitations and Windows run instructions.
- `data/observations.csv`: normalized public observations.
- `data/raw/`: unmodified CBK and KNBS source documents with acquisition metadata and checksums.
- `data/sources.json`: provenance and explicit reconstruction departures.
- `results/`: forecasts, benchmarks, news attribution, model parameters, diagnostics and validation results.
- The original 68-page working paper is included as a PDF.

Download or clone the repository, then open `report.html` locally. GitHub's source viewer does not render this local HTML report as a website.

## Reproduce

Python package versions are pinned in `requirements.txt`. In an environment with those dependencies available, run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -B download_data.py
python -B replicate.py
python -B validate_results.py
```

`download_data.py` uses saved sources when their checksums match. Use `--refresh` only when intentionally reacquiring revised source data. `replicate.py` runs offline from the normalized observations. The default exercise uses one, two and three factors across 103 forecast dates, producing 618 quarterly predictions and calendar-year forecasts.

## Results and validation

All 54 quarterly model fits converged, and seven focused behavior tests passed. Saved validation also checks source hashes, release-masked estimation samples, news reconciliation, forecast metrics and report assets.

For the reconstructed two-factor model, nowcast RMSE is approximately **1.38 percentage points** versus **0.83** for an AR(1) benchmark; backcast RMSE is approximately **1.16** versus **0.63**. Thus this public-data exercise does **not** reproduce the paper's favorable comparative performance. These differences do not establish that the published results are incorrect, since the inputs and vintages differ.

Forecast errors use percentage points of 100 times log year-on-year GDP growth. Conditional prediction intervals do not include parameter or revision uncertainty.

## Sources and rights

Original paper: <https://www.imf.org/-/media/files/publications/wp/2026/english/wpiea2026032-source-pdf.pdf>

Public inputs come from the Central Bank of Kenya and Kenya National Bureau of Statistics. Exact URLs, acquisition timestamps, table locations and checksums are included in the source metadata. Third-party publications and data remain subject to their respective owners' terms; their inclusion does not imply a new license or affiliation with the authors or the IMF.

Machine-local file inventories, runtime caches and temporary authentication material are excluded from Git.
