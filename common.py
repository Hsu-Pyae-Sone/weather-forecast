"""
Shared helpers used by every step script. Read this file once; the steps import from it.

Dataset: daily MINIMUM temperature (°C) in Helsinki, Finland, 2015-2024.
  - Source: Finnish Meteorological Institute (FMI) open data, licence CC BY 4.0
  - Station: Helsinki Kaisaniemi (city centre), FMI station id "fmisid" 100971
Task:    predict TOMORROW's minimum temperature from the days before it (regression).
"""
# Lets us change how text is printed to the terminal (see the reconfigure call below).
import sys
# Python's built-in HTTP client, used to download the data (like fetch() in JavaScript).
import urllib.request
# Python's built-in XML parser; ET is the usual short name. FMI's API answers in XML.
import xml.etree.ElementTree as ET
# Path builds file paths that work on both Windows and Mac/Linux.
from pathlib import Path

# NumPy: fast math on whole arrays of numbers at once.
import numpy as np
# pandas: tables in code. A DataFrame is a whole table; a Series is one labelled column.
import pandas as pd
# plotext draws charts as text inside the terminal, so no chart window has to open.
import plotext as plt
# Ready-made formulas from scikit-learn (the main ML library) that measure prediction error.
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

sys.stdout.reconfigure(encoding="utf-8")  # so °C and the terminal charts print on Windows too

# The folder this file lives in; every other path is built from it, so scripts run from anywhere.
ROOT = Path(__file__).parent
# Where the downloaded data is cached, so we only download it once.
DATA_PATH = ROOT / "data" / "helsinki-daily-min-temperatures.csv"
# Where step 4 saves the trained model and where steps 5 and 6 load it from.
MODEL_PATH = ROOT / "models" / "gb_model.joblib"

# FMI's open API (no key needed). It returns XML and allows about one year per request.
# The API's base address and the kind of request we are making.
FMI_URL = ("https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature"
           # Which ready-made query to run: daily weather observations.
           "&storedquery_id=fmi::observations::weather::daily::simple"
           # Which station (Helsinki Kaisaniemi) and which measurement (tmin = daily minimum).
           "&fmisid=100971&parameters=tmin"
           # The date range; {year} is a placeholder that .format() fills in for each request.
           "&starttime={year}-01-01T00:00:00Z&endtime={year}-12-31T00:00:00Z")
YEARS = range(2015, 2025)  # 2015 ... 2024 = 10 full years

# Time-series split: we cut by DATE, never randomly (the future must stay hidden).
#   2015-2020 -> train       (the model learns from this)
#   2021-2022 -> validation  (we tune settings with this)
#   2023-2024 -> test        (touched only at the very end, like real "future" days)
# The last day of the training period.
TRAIN_END = "2020-12-31"
# The last day of the validation period; everything after it is the test set.
VAL_END = "2022-12-31"

# The input columns ("features") the model sees. The answer column is "target".
FEATURES = ["lag_1", "lag_2", "lag_3", "lag_7", "roll_mean_7", "month", "doy_sin", "doy_cos"]


# ----------------------------------------------------------------------------- data
# Downloads the raw data. Only runs when the CSV file does not exist yet.
def download():
    """Fetch the data from FMI, one year per request, and save it as a simple CSV (Date, Temp)."""
    # XML tag prefix -> full schema name. The XML search below needs this to recognise the tags.
    ns = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
    # Collects one (date, temperature) pair per day.
    rows = []
    # One request per year, because the API limits how much it returns at once.
    for year in YEARS:
        # Shows progress, because the download takes a few seconds.
        print(f"Downloading {year} from FMI open data ...")
        # Fills the year into the URL, calls the API and reads the XML reply (60-second limit).
        xml = urllib.request.urlopen(FMI_URL.format(year=year), timeout=60).read()
        for el in ET.fromstring(xml).iterfind(".//BsWfs:BsWfsElement", ns):  # one element = one day
            date = el.findtext("BsWfs:Time", namespaces=ns)[:10]             # "2015-01-01T00:00:00Z"
            rows.append((date, el.findtext("BsWfs:ParameterValue", namespaces=ns)))  # "NaN" if missing
    # Creates the data/ folder if it isn't there yet (exist_ok = no error if it already is).
    DATA_PATH.parent.mkdir(exist_ok=True)
    # Turns the pairs into a table and saves it as CSV; index=False skips pandas' row numbers.
    pd.DataFrame(rows, columns=["Date", "Temp"]).to_csv(DATA_PATH, index=False)


# Gives callers the data without them caring whether it was downloaded or cached.
def load_raw():
    """Return the raw data table (downloaded from FMI on the first run only)."""
    # A simple cache check: no file yet means this is the first run...
    if not DATA_PATH.exists():
        # ...so download the file once.
        download()
    # Reads the CSV into a table; parse_dates turns the Date text into real date values.
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


# Fixes data problems first, because models cannot handle gaps in the data.
def clean(raw):
    """Raw table -> one clean value per calendar day (a pandas Series indexed by date)."""
    # Keeps only the Temp column, labels each value by its date, and makes sure they are numbers.
    s = raw.set_index("Date")["Temp"].astype(float)
    s = s.asfreq("D")        # insert a row for every calendar day; missing days become NaN
    return s.interpolate()   # fill each gap with a straight line between its neighbours


# Creates the model's inputs, since the CSV only has dates and temperatures ("feature engineering").
def make_features(s):
    """Turn ONE column of temperatures into a TABLE the model can learn from.

    Each row = one day to predict ("target"). Every feature only looks at days BEFORE it,
    otherwise the model would peek at the answer (called "data leakage").
    """
    # Starts the table with the answer column: the temperature to predict on each date.
    df = pd.DataFrame({"target": s})
    # Adds one "lag" column each for 1, 2, 3 and 7 days back.
    for lag in (1, 2, 3, 7):
        df[f"lag_{lag}"] = s.shift(lag)                 # temperature `lag` days earlier
    df["roll_mean_7"] = s.shift(1).rolling(7).mean()    # average of the previous 7 days
    df["month"] = s.index.month                         # 1..12, captures the season
    doy = s.index.day_of_year                             # 1..366
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)    # day-of-year on a circle, so that
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)    # Dec 31 ends up right next to Jan 1
    return df.dropna()                                  # first 7 days lack full history


# Divides the table by date, so the model is always tested on days later than it learned from.
def split(df):
    """Chronological split -> (train, val, test) DataFrames."""
    # Training rows: everything up to the end of 2020.
    train = df[df.index <= TRAIN_END]
    # Validation rows: 2021 and 2022.
    val = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    # Test rows: everything after 2022.
    test = df[df.index > VAL_END]
    # Returns all three parts so the caller can unpack them in one line.
    return train, val, test


# scikit-learn expects inputs and answers separately, so this splits one table into two.
def xy(part):
    """Separate a table into inputs X and answers y."""
    # X = just the feature columns (the question); y = the target column (the answer).
    return part[FEATURES], part["target"]


# A convenience wrapper so each step script can get its data in a single line.
def load_splits():
    """The whole data pipeline in one call: raw -> clean -> features -> split."""
    # Reads from the inside out: load, clean, build features, then split.
    return split(make_features(clean(load_raw())))


# ----------------------------------------------------------------------------- metrics
# Grades a set of predictions. Takes the true answers and the model's guesses.
def scores(y_true, y_pred):
    """MAE / RMSE in °C (lower is better) and R² (1.0 = perfect, 0 = no better than the mean)."""
    # Returns the three grades as a dictionary (like a JS object).
    return {
        # Mean Absolute Error: the average miss, ignoring whether it was too high or too low.
        "MAE": mean_absolute_error(y_true, y_pred),
        # Root Mean Squared Error: like MAE, but big misses count extra.
        "RMSE": root_mean_squared_error(y_true, y_pred),
        # R-squared: how much of the ups and downs in temperature the model explains.
        "R2": r2_score(y_true, y_pred),
    }


# Prints several models' scores in columns, so they are easy to compare.
def print_scores(rows):
    """rows = {"name": scores_dict}. Prints a small aligned table."""
    # The header row. :<28 pads left-aligned to 28 characters; :>9 right-aligns to 9.
    print(f"{'':<28}{'MAE °C':>9}{'RMSE °C':>10}{'R²':>8}")
    # Goes through each model's name and its scores dictionary.
    for name, s in rows.items():
        # One row of the table; .3f shows 3 decimals. The widths match the header above.
        print(f"{name:<28}{s['MAE']:>9.3f}{s['RMSE']:>10.3f}{s['R2']:>8.3f}")


# ----------------------------------------------------------------------------- terminal output
# Prints a section heading, so each part of a step's long output is easy to find.
def banner(text):
    # A blank line, then the text between two lines of 78 "=" characters.
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# The shared line chart helper, so every step's line charts look the same.
def plot_lines(lines, title, xlabel="", ylabel="", dates=None):
    """Draw line chart(s) in the terminal. lines = {"legend label": list_of_y_values}.
    If `dates` is given, the x-axis is labelled with years instead of row numbers."""
    # plotext remembers the previous chart, so wipe it or the charts would stack up.
    plt.clear_figure()
    plt.theme("clear")  # use the terminal's own background (works in dark and light themes)
    # The chart size in terminal characters: 100 wide, 22 tall.
    plt.plotsize(100, 22)
    # Draws one line per entry, which lets one chart compare e.g. "actual" and "predicted".
    for label, y in lines.items():
        # marker="braille" uses tiny dot characters, which gives a smoother line.
        plt.plot(list(y), label=label, marker="braille")
    # Only relabel the x-axis if the caller passed dates.
    if dates is not None:
        # Finds the row position of every January 1st.
        first_of_year = [i for i, d in enumerate(dates) if d.day_of_year == 1]
        # Puts a tick at each of those positions, labelled with its year.
        plt.xticks(first_of_year, [str(dates[i].year) for i in first_of_year])
    # Sets the heading above the chart.
    plt.title(title)
    # Sets the label for the horizontal axis.
    plt.xlabel(xlabel)
    # Sets the label for the vertical axis.
    plt.ylabel(ylabel)
    # Actually draws the chart in the terminal.
    plt.show()


# The shared bar chart helper, used to compare values such as months or features.
def plot_bars(labels, values, title):
    """Horizontal bar chart in the terminal."""
    # Wipes the previous chart so it doesn't get mixed into this one.
    plt.clear_figure()
    # Uses the terminal's own background, same as plot_lines.
    plt.theme("clear")
    plt.plotsize(100, len(labels) + 4)  # +4 rows for title and axis -> exactly one row per bar
    # plotext draws the first bar at the bottom, so reverse to read top-down
    plt.bar([str(l) for l in labels][::-1], list(values)[::-1], orientation="horizontal", width=0.5)
    # Sets the heading above the chart.
    plt.title(title)
    # Actually draws the chart in the terminal.
    plt.show()


# The shared histogram helper. A histogram counts how many values land in each range.
def plot_hist(values, title, bins=30):
    # Wipes the previous chart so it doesn't get mixed into this one.
    plt.clear_figure()
    # Uses the terminal's own background, same as plot_lines.
    plt.theme("clear")
    # The chart size in terminal characters: 100 wide, 20 tall.
    plt.plotsize(100, 20)
    # Sorts the values into `bins` equal-width ranges and draws one bar per range.
    plt.hist(list(values), bins)
    # Sets the heading above the chart.
    plt.title(title)
    # Actually draws the chart in the terminal.
    plt.show()
