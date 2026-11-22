"""Acquire and normalize official public inputs, preserving original documents."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from pypdf import PdfReader
import requests

from data_model import SERIES
from file_utils import ROOT, output_session, write_bytes, write_json, write_text

DATA = ROOT / "data"
RAW = DATA / "raw"
SESSION = requests.Session()
MONTHS = {name.lower(): number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
MONTHS.update({name[:3]: value for name, value in list(MONTHS.items())})


def acquire(name, url, refresh=False, post=None):
    path = RAW / name
    meta = RAW / (name + ".meta.json")
    if path.exists() and meta.exists() and not refresh:
        record = json.loads(meta.read_text(encoding="utf-8"))
        content = path.read_bytes()
        if record["sha256"] != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Cached source checksum mismatch: {name}")
        return content
    for attempt in range(3):
        try:
            response = SESSION.post(url, data=post, timeout=90) if post else SESSION.get(url, timeout=90)
            response.raise_for_status()
            content = response.content
            if name.endswith(".pdf") and not content.startswith(b"%PDF"):
                raise ValueError(f"Not a PDF: {url}")
            write_bytes(path, content)
            write_json(meta, {"url": url, "method": "POST (read-only table query)" if post else "GET",
                              "post_data": post, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                              "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
            print(f"Acquired {name}: {len(content):,} bytes", flush=True)
            return content
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def acquire_all(refresh=False):
    sources = {
        "remittances.html": "https://www.centralbank.go.ke/diaspora-remittances/",
        "mobile_payments.html": "https://www.centralbank.go.ke/national-payments-system/mobile-payments/",
        "foreign_trade.html": "https://www.centralbank.go.ke/foreign-trade-summary/",
        "m2.csv": "https://www.centralbank.go.ke/uploads/monetary_and_finance_statistics/377062609_Depository%20Corporation%20Survey%20-%20CSV.csv",
        "gdp_rebased_2021.pdf": "https://www.knbs.or.ke/wp-content/uploads/2021/09/Revision-and-Rebasing-of-National-Accounts-Report.pdf",
        "gdp_2025q3.pdf": "https://www.knbs.or.ke/wp-content/uploads/2025/12/Kenya-Quarterly-Gross-Domestic-Product-Third-Quarter-2025.pdf",
        "lei_archive.json": "https://www.knbs.or.ke/wp-json/wp/v2/media?search=Leading%20Economic%20Indicators%20December&per_page=100",
    }
    for name, url in sources.items():
        acquire(name, url, refresh)
    acquire("principal_exports.json", "https://www.centralbank.go.ke/wp-admin/admin-ajax.php?action=get_wdtable&table_id=27",
            refresh, post={"draw": 1, "start": 0, "length": -1})
    archive = json.loads((RAW / "lei_archive.json").read_bytes())
    for entry in archive:
        url = entry.get("source_url", "")
        match = re.search(r"December-(20\d{2})\.pdf", url, re.I)
        if match and 2009 <= int(match[1]) <= 2025:
            acquire(f"lei_{match[1]}.pdf", url, refresh)


def numeric(value):
    value = str(value).strip().replace(",", "").replace(" ", "").replace("\u00a0", "")
    if value in {"", "-", "..", "...", "n.a", "nan", "None"}:
        return np.nan
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    return float(value)


def monthly_period(year, month):
    month = str(month).strip().lower()
    number = int(month) if month.isdigit() else MONTHS[month[:3]]
    return f"{int(year):04d}-{number:02d}"


def parse_row_year(rest, current_year, edition):
    match = re.match(r"^-?\s*(\d{2}|20\d{2})(?![\d,.])\s+", rest)
    if match:
        token = int(match[1])
        candidate = token + 2000 if token < 100 else token
        if candidate in (edition - 1, edition) or rest.startswith("-"):
            return candidate, rest[match.end():]
    return current_year, rest


def gdp_rows(path, page_index, start, expected):
    text = PdfReader(path).pages[page_index].extract_text()
    values = []
    for line in text.splitlines():
        if re.match(r"^[1-4]\s", line.strip()):
            cells = re.findall(r"\(?-?[\d,]+(?:\.\d+)?\)?", line)
            if len(cells) > 8:
                values.append(numeric(cells[-2]))
    if len(values) != expected:
        raise ValueError(f"GDP table {path.name}: expected {expected} quarters, found {len(values)}")
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="Q"))


def normalize():
    records = []
    provenance = []

    def add(key, period, value, source, detail=""):
        value = numeric(value)
        if np.isfinite(value):
            records.append({"series": key, "period": str(period), "value": value,
                            "source": source, "detail": detail})

    rem = pd.read_html(StringIO((RAW / "remittances.html").read_text(encoding="utf-8")), displayed_only=False)[0]
    for row in rem.itertuples(index=False, name=None):
        add("remittances", monthly_period(row[0], row[1]), numeric(row[-1]) / 1000,
            "remittances.html", "Total remittances, thousands USD / 1000 = million USD")
    mobile = pd.read_html(StringIO((RAW / "mobile_payments.html").read_text(encoding="utf-8")), displayed_only=False)[0]
    for row in mobile.itertuples(index=False, name=None):
        add("mobile_money", monthly_period(row[0], row[1]), row[5], "mobile_payments.html", "Agent cash in/cash out value, billion KES")
    trade = pd.read_html(StringIO((RAW / "foreign_trade.html").read_text(encoding="utf-8")), displayed_only=False)[0]
    for row in trade.itertuples(index=False, name=None):
        period = monthly_period(row[0], row[1])
        add("exports", period, row[7], "foreign_trade.html", "Total FOB including re-exports, million KES")
        add("imports", period, row[4], "foreign_trade.html", "Total CIF imports, million KES")
    principal = json.loads((RAW / "principal_exports.json").read_bytes())["data"]
    for row in principal:
        period = monthly_period(row[0], row[1])
        add("coffee", period, row[2], "principal_exports.json", "Coffee volume, tonnes")
        add("tea", period, row[5], "principal_exports.json", "Tea volume, tonnes")
    old = gdp_rows(RAW / "gdp_rebased_2021.pdf", 63, "2009Q1", 48)
    recent = gdp_rows(RAW / "gdp_2025q3.pdf", 11, "2020Q1", 23)
    for quarter, value in old.loc[:"2019Q4"].items():
        add("gdp", quarter, value, "gdp_rebased_2021.pdf", "PDF p64, Annex 14; GDP at market prices, constant 2016 KES million")
    for quarter, value in recent.items():
        add("gdp", quarter, value, "gdp_2025q3.pdf", "PDF p12, Table 2; GDP at market prices, constant 2016 KES million")
    assert old.iloc[0] == 1335552 and recent.iloc[-1] == 2762025
    normalize_m2(add)
    normalize_lei(add)
    frame = pd.DataFrame(records)
    result = []
    for key, spec in SERIES.items():
        rows = frame.loc[frame.series == key].copy()
        frequency = "Q" if key == "gdp" else "M"
        if not len(rows):
            continue
        periods = pd.PeriodIndex(rows.period, freq=frequency)
        start = pd.Period("2009Q1" if key == "gdp" else "2009-01", freq=frequency)
        end = pd.Period(spec["end"], freq=frequency)
        rows = rows.loc[(periods >= start) & (periods <= end)]
        if rows.period.duplicated().any():
            raise ValueError(f"Duplicate final values for {key}")
        result.append(rows)
    combined = pd.concat(result, ignore_index=True).sort_values(["series", "period"])
    write_text(DATA / "observations.csv", combined.to_csv(index=False, float_format="%.10g"))
    source_records = []
    for meta in sorted(RAW.glob("*.meta.json")):
        entry = json.loads(meta.read_text(encoding="utf-8"))
        entry["local_file"] = meta.name.removesuffix(".meta.json")
        source_records.append(entry)
    write_json(DATA / "sources.json", {
        "sources": source_records,
        "series": SERIES,
        "departures": [
            "PMI omitted: no complete free official historical series verified; S&P historical data are subscription-based.",
            "GDP splices the September 2021 rebasing report for 2009–2019 with December 2025 quarterly release for 2020–2025Q3. Both use constant 2016 prices, but this is mixed vintage.",
            "Visitor arrivals use a consistent JKIA+MIA airport proxy, excluding other borders even when those are available.",
            "Electricity uses domestic/local generation, excluding imports; the paper does not identify the precise column.",
            "Monthly KNBS PDF data use the newest December edition containing each month; original revision vintages are unavailable.",
            "March/April 2012 CBK trade gaps are repaired from KNBS December 2012 LEI, Table 12, PDF page 25.",
            "All inputs are truncated to Table 1 endpoints; historical public files are saved unmodified.",
            "Electricity July–September 2009 is left missing: the December 2009 table labels these rows as 2010. No date correction or interpolation is inferred.",
            "Airport arrivals for October 2010 are absent from the acquired December editions and remain missing.",
        ],
        "normalized_coverage": combined.groupby("series").agg(first=("period", "min"), last=("period", "max"), n=("value", "size")).reset_index().to_dict("records"),
    })
    print(combined.groupby("series").agg(first=("period", "min"), last=("period", "max"), n=("value", "size")).to_string(), flush=True)


def normalize_m2(add):
    import csv
    rows = list(csv.reader((RAW / "m2.csv").read_text(encoding="utf-8-sig").splitlines()))
    assert rows[33][1].strip() == "Money and Quasi-money (M2)"
    assert rows[2][3].strip() == "1997" and rows[3][3].strip() == "Dec"
    dates = pd.period_range("1997-12", periods=len(rows[3]) - 3, freq="M")
    for column, date in enumerate(dates, 3):
        month = rows[3][column].strip()[:3].lower()
        if MONTHS.get(month) != date.month:
            raise ValueError(f"Unexpected monetary survey month sequence at column {column}")
        add("m2", date, rows[33][column], "m2.csv",
            "Physical row 34, M2, standard survey, million KES; contiguous month sequence verified from Dec 1997 anchor")
    if len(dates) < 300:
        raise ValueError(f"Unexpected monetary survey coverage: {len(dates)}")


def normalize_lei(add):
    pages = {
        2009: (30, 38, 41), 2010: (31, 39, 42), 2011: (32, 40, 43),
        2012: (30, 38, 41), 2013: (31, 40, 43), 2014: (29, 40, 43),
        2015: (29, 40, 43), 2016: (33, 46, 50), 2017: (31, 41, 44),
        2018: (37, 48, 52), 2019: (33, 44, 48), 2020: (33, 44, 48),
        2021: (33, 44, 48), 2022: (32, 41, 44), 2023: (28, 35, 37),
        2024: (28, 35, 38), 2025: (32, 40, 43),
    }
    selected = {}
    source_issues = []
    month_pattern = re.compile(r"^\s*(January|February|March|April|May|June|July|August|September|October|November|December|Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", re.I)

    def values_in(text):
        return [numeric(value) for value in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]

    for edition, positions in pages.items():
        reader = PdfReader(RAW / f"lei_{edition}.pdf")
        for key, page_number in zip(("electricity", "vehicles", "arrivals"), positions):
            text = reader.pages[page_number - 1].extract_text()
            phrase = {"electricity": "localelectricitygenerationbysource",
                      "vehicles": "productionofassembledvehicles",
                      "arrivals": "visitorarrivals"}[key]
            heading = re.search(r"\s*".join(phrase), text, re.I)
            if heading is None:
                raise ValueError(f"Missing {key} heading in LEI {edition} page {page_number}")
            text = text[heading.end():]
            text = re.split(r"\b(?:Table\s+\d|Fig(?:ure)?\s*\d|Source\s*:)", text, maxsplit=1, flags=re.I)[0]
            current_year = None
            column_years = []
            parsed = 0
            for line in text.splitlines():
                year_match = re.fullmatch(r"\s*(20\d{2})\s*\*?\s*", line)
                if year_match:
                    current_year = int(year_match[1])
                    continue
                if key == "vehicles":
                    years = re.findall(r"\b20\d{2}\b", line)
                    if len(years) >= 3:
                        column_years = list(map(int, years))
                        continue
                match = month_pattern.match(line)
                if not match:
                    continue
                month = MONTHS[match[1].lower()[:3]]
                rest = line[match.end():].strip()
                if key == "vehicles":
                    if not column_years:
                        raise ValueError(f"Missing vehicle column years: LEI {edition}")
                    cells = values_in(rest)
                    for year, value in zip(column_years, cells):
                        if 2009 <= year <= edition and np.isfinite(value):
                            period = f"{year}-{month:02d}"
                            selected[(key, period)] = (value, f"lei_{edition}.pdf", f"PDF p{page_number}, assembled vehicle production, units")
                            parsed += 1
                    continue
                year, rest = parse_row_year(rest, current_year, edition)
                current_year = year
                if year is None:
                    continue
                if not 2009 <= year <= edition:
                    if year > edition:
                        source_issues.append({"file": f"lei_{edition}.pdf", "page": page_number, "series": key,
                                              "row": line.strip(), "action": "Excluded date outside edition year; no inferred correction."})
                    continue
                cells = values_in(rest)
                if len(cells) < (4 if key == "electricity" else 2):
                    continue
                value = cells[-1] if key == "electricity" else cells[0] + cells[1]
                period = f"{year}-{month:02d}"
                detail = f"PDF p{page_number}, " + ("local generation total, million kWh (excluding imports)" if key == "electricity" else "JKIA + MIA, persons; airport-only proxy")
                selected[(key, period)] = (value, f"lei_{edition}.pdf", detail)
                parsed += 1
            print(f"Parsed LEI {edition} {key}: {parsed} source cells", flush=True)
            if parsed < 8:
                raise ValueError(f"Insufficient {key} extraction in LEI {edition}: {parsed}")
    for (key, period), (value, source, detail) in sorted(selected.items()):
        add(key, period, value, source, detail)
    repair_text = PdfReader(RAW / "lei_2012.pdf").pages[24].extract_text()
    for period, exports, imports in [("2012-03", 44362.36, 124621.83), ("2012-04", 40039.70, 102765.50)]:
        for key, value in [("exports", exports), ("imports", imports)]:
            if f"{value:,.2f}" not in repair_text:
                raise ValueError(f"Trade repair value not confirmed in source: {value}")
            add(key, period, value, "lei_2012.pdf", "PDF p25, Table 12; repairs missing CBK table month")
    write_json(DATA / "source_quality_issues.json", source_issues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    with output_session():
        acquire_all(args.refresh)
        if not args.download_only:
            normalize()
