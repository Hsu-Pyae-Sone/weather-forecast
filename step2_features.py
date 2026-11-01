"""
STEP 2 - Data processing & transformation: one column of numbers -> a table a model can learn.

A model like gradient boosting needs rows of  (inputs X  ->  answer y).
Our CSV only has (date, temperature). So we CREATE inputs from the past ("features").
Run:  python step2_features.py
"""
# NumPy, for the sin/cos math and the distance calculation below.
import numpy as np
# pandas, for building the small demo tables.
import pandas as pd

# The shared helpers: the feature list, the data pipeline functions and the printing tools.
from common import FEATURES, banner, clean, load_raw, make_features, plot_lines, split

s = clean(load_raw())  # step 1's cleaning: 3,652 days, no gaps

# ----- Part A: turn past days into input columns -----
banner("A. Sliding window: yesterday, 2 days ago, ... become the INPUT columns")
# Labels the "before" view.
print("Raw series (first 10 days):")
# Shows the starting point: just a date and a temperature.
print(s.head(10).to_string(), "\n")
# shift(n) moves the values down n rows, so each row can see the value from n days earlier.
demo = pd.DataFrame({"target (today)": s, "lag_1": s.shift(1), "lag_2": s.shift(2), "lag_3": s.shift(3)})
# Explains what to look for in the table below.
print("After shift(): each row now holds today's answer AND the previous days as inputs.")
# Explains the empty cells: the first days have no earlier days to look back at.
print("NaN = 'no history yet' -> those first rows get dropped.\n")
# Shows the first 6 rows, where the values visibly step down and to the right.
print(demo.head(6).to_string())

# ----- Part B: the rolling average -----
banner("B. Rolling mean: the average of the 7 days BEFORE today (smooths out noise)")
# Explains why the code below calls shift(1) before rolling(7).
print("shift(1) first, so today's own temperature is never part of its own input (no leakage).")
# Builds a demo table of each day's answer next to the average of the 7 days before it.
roll = pd.DataFrame({"target": s, "roll_mean_7": s.shift(1).rolling(7).mean()})
# Shows rows 6-9: row 6 is still empty, and row 7 is the first with 7 days of history.
print(roll.iloc[6:10].round(2).to_string())
# Recalculates row 7 by hand, proving the rolling average does what we claim.
print(f"Check row {s.index[7].date()}: mean of the 7 days before = {s.iloc[0:7].mean():.2f}")

# ----- Part C: tell the model the time of year -----
banner("C. Calendar features: tell the model WHERE in the year we are")
# Four sample dates, including Dec 31 and Jan 1, which are neighbours in real life.
days = pd.to_datetime(["2017-01-01", "2017-04-01", "2017-07-01", "2017-12-31"])
# Converts each date to its day number within the year (Jan 1 = 1, Dec 31 = 365).
doy = days.day_of_year
# Builds a table comparing the plain day number...
cal = pd.DataFrame({"day_of_year": doy,
                    # ...with its position on a circle: the sine part...
                    "doy_sin": np.sin(2 * np.pi * doy / 365.25).round(3),
                    # ...and the cosine part, using the dates as row labels.
                    "doy_cos": np.cos(2 * np.pi * doy / 365.25).round(3)}, index=days.date)
# Shows the comparison table.
print(cal.to_string())
# Explains the problem with plain day numbers: the year's end looks far from its start.
print("\nPlain day_of_year says Dec 31 (365) and Jan 1 (1) are FAR apart. In reality they are")
# Introduces the fix, which the next statement proves with a number.
print("neighbours. Putting the day on a circle (sin, cos) fixes that:")
# Prints the label for the distance...
print("  distance(Dec 31, Jan 1) in (sin,cos) space =",
      # ...then the value: Dec 31 (row 3) minus Jan 1 (row 0); hypot gives the straight-line distance.
      round(float(np.hypot(*(cal.iloc[3, 1:] - cal.iloc[0, 1:]))), 3))

# ----- Part D: the finished table -----
banner("D. The final feature table (what the model actually sees)")
# Runs the real feature code from common.py (parts A-C were only demos).
table = make_features(s)
# Shows the first rows of the finished table.
print(table.head().round(2).to_string())
# Reports the table's size: the number of rows and input columns, plus the answer column.
print(f"\nShape: {table.shape[0]} rows x {len(FEATURES)} features + 1 target")
# Scaling means rescaling columns to a similar range, which many other model types need.
print("Note: NO scaling/normalising needed. Trees split on thresholds like 'lag_1 <= 12.3',")
# Explains why: a tree only asks "is it above this value?", so the size of the units doesn't matter.
print("so multiplying a column by 100 would not change the model at all.")

# ----- Part E: split into train, validation and test -----
banner("E. Train / validation / test split - by TIME, never shuffled")
# Cuts the table into the three date ranges defined in common.py.
train, val, test = split(table)
# Goes through the three parts, each with a display name.
for name, part in [("train", train), ("validation", val), ("test", test)]:
    # Prints each part's name and date range...
    print(f"  {name:<11} {part.index.min().date()} -> {part.index.max().date()}  "
          # ...then its row count and its share of the whole table (:.0% formats it as a percent).
          f"{len(part):>5} rows ({len(part) / len(table):.0%})")
# A safety check that stops the script if the three date ranges ever overlap or fall out of order.
assert train.index.max() < val.index.min() and val.index.max() < test.index.min()
# Prints a timeline of the split and explains why it isn't random.
print("""
  |-------- train (learn) --------|-- validation (tune) --|---- test (final exam) ----|
  2015                          2021                     2023                      2024
Why not a random split? Shuffling would put 2024 days in training and 2017 days in the test,
so the model would 'remember the future'. Real forecasts only ever know the past.""")

# The answer column for all dates, which the chart below divides into three lines.
full = table["target"]
# .where() keeps values that match and blanks the rest, so this line only covers training dates.
plot_lines({"train": full.where(full.index <= train.index.max()),
            # The same idea, keeping only the validation dates.
            "validation": full.where((full.index >= val.index.min()) & (full.index <= val.index.max())),
            # The same idea, keeping only the test dates.
            "test": full.where(full.index >= test.index.min())},
           # The chart title and axis labels, with years on the x-axis.
           "Target temperature, coloured by split", "year", "°C", dates=table.index)
