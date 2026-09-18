"""
download_real_data.py

Downloads and prepares real-world COVID-19 case-count data for the
first-wave validation experiment (Italy as the primary source, South
Korea as a secondary, independent source), from the Johns Hopkins
University CSSE COVID-19 data repository.

DATA SOURCE AND AN IMPORTANT CAVEAT
--------------------------------------
Source: Johns Hopkins University Center for Systems Science and
Engineering (JHU CSSE) COVID-19 Data Repository
(https://github.com/CSSEGISandData/COVID-19), cumulative confirmed cases,
country level.

IMPORTANT: this repository was archived (set to read-only) by JHU CSSE on
March 10, 2023, and is no longer updated. It should be cited as an
archived historical source, not as a live data feed. The historical data
used here (January 2020 through mid-2020, covering the first pandemic
wave) remain unaffected by the archival and are the correct, complete
historical record for that period.

Both Italy and South Korea are drawn from the SAME JHU-aggregated source
for consistency of format and processing, even though the original
project plan considered a separate, South-Korea-specific dataset
(DS4C/KCDC). South Korea is retained as the secondary series specifically
because its early, extensive testing capacity produced comparatively
complete case ascertainment relative to many other countries during the
first wave -- providing a natural contrast to Italy's more constrained
testing capacity in February-March 2020, without requiring a
differently-formatted secondary data source.

MODELING ASSUMPTION (please read before fitting)
-----------------------------------------------------
This script prepares CUMULATIVE CONFIRMED CASE COUNTS. When fitting the
SEIR model to this series (see real_data_fit_nls.py), cumulative
confirmed cases are treated as a noisy, underreported proxy for the
cumulative number of individuals who have ever left the Exposed
compartment, i.e., observed(t) ~ rho * (I(t) + R(t)), where rho is an
unknown reporting/ascertainment fraction estimated jointly with the
epidemiological parameters. This is a simplifying assumption (real
reporting involves delays, changing testing policy over time, and
asymptomatic cases that may never be tested) -- it is the standard
simplification used throughout this benchmark's real-data validation, and
its limitations should be discussed explicitly in the manuscript.

HOW TO RUN
-----------
    python download_real_data.py

Requires internet access to raw.githubusercontent.com. Writes
real_data_italy.csv and real_data_korea.csv to the current directory,
each with columns: date, day, cumulative_confirmed.

DEPENDENCIES
-------------
pandas, requests (or urllib, no extra dependency beyond pandas' built-in
CSV reading over HTTP)

LICENSE
--------
This script is intended to be released alongside the associated publication
under an open license (e.g., MIT) to support reproducibility. The
downloaded data itself remains subject to JHU CSSE's data license
(CC BY 4.0) -- see https://github.com/CSSEGISandData/COVID-19 for the
required attribution.
"""

import pandas as pd

JHU_CONFIRMED_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

# First-wave windows, chosen to start a few days before the country's case
# count begins sustained growth and to extend well past the first peak.
FIRST_WAVE_WINDOWS = {
    "italy": {"country": "Italy", "start_date": "2/20/20", "end_date": "6/20/20"},
    "korea": {"country": "Korea, South", "start_date": "1/22/20", "end_date": "5/22/20"},
}


def download_and_prepare(country_key):
    config = FIRST_WAVE_WINDOWS[country_key]
    print(f"Downloading JHU CSSE confirmed-case data for {config['country']}...")

    full_df = pd.read_csv(JHU_CONFIRMED_URL)
    country_row = full_df[full_df["Country/Region"] == config["country"]]
    if len(country_row) == 0:
        raise ValueError(f"Country '{config['country']}' not found in the JHU dataset.")
    # Some countries have multiple province/state rows; sum them for a
    # national total (Italy and South Korea each have exactly one row, so
    # this is a no-op for them, but it is done generally for robustness).
    date_columns = full_df.columns[4:]
    national_series = country_row[date_columns].sum(axis=0)

    dates = pd.to_datetime(national_series.index, format="%m/%d/%y")
    series_df = pd.DataFrame({
        "date": dates,
        "cumulative_confirmed": national_series.values,
    }).sort_values("date").reset_index(drop=True)

    start = pd.to_datetime(config["start_date"], format="%m/%d/%y")
    end = pd.to_datetime(config["end_date"], format="%m/%d/%y")
    windowed = series_df[(series_df["date"] >= start) & (series_df["date"] <= end)].copy()
    windowed["day"] = (windowed["date"] - windowed["date"].iloc[0]).dt.days
    windowed = windowed[["date", "day", "cumulative_confirmed"]].reset_index(drop=True)

    output_path = f"real_data_{country_key}.csv"
    windowed.to_csv(output_path, index=False)
    print(f"Saved {len(windowed)} days of data to {output_path} "
          f"(from {windowed['date'].iloc[0].date()} to {windowed['date'].iloc[-1].date()}, "
          f"final cumulative count: {int(windowed['cumulative_confirmed'].iloc[-1])})")
    return windowed


def main():
    for country_key in FIRST_WAVE_WINDOWS:
        download_and_prepare(country_key)


if __name__ == "__main__":
    main()
