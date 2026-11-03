"""
STEP 1 - Look at the data before touching any model.

Questions we answer here:
  1. What does one row look like?          3. Is there a trend or a yearly cycle?
  2. Are any days missing or broken?       4. Does yesterday tell us about today?
Run:  python step1_explore.py
"""
# Pulls in the shared helpers: heading, data loading and cleaning, and the terminal charts.
from common import banner, clean, load_raw, plot_bars, plot_lines

# ----- Question 1: what does the data look like? -----
banner("1. The raw data: one row = one day, one number = that day's minimum temperature")
# Loads the table of dates and temperatures (downloads it on the first run).
raw = load_raw()
# Shows the first 5 rows, a quick look at the real data before trusting it.
print(raw.head(), "\n")
# Shows the table's size and column names.
print(f"Rows: {len(raw)}   Columns: {list(raw.columns)}")
# Shows the first and last date, confirming we got the period we expected.
print(f"From {raw['Date'].min().date()} to {raw['Date'].max().date()}\n")
print(raw["Temp"].describe().round(2))  # count, mean, std (spread), min, quartiles, max

# ----- Question 2: is anything missing or broken? -----
banner("2. Data quality: are any calendar days missing?")
s = clean(raw)  # inserts the missing days and fills them in
# Finds the dates that exist after cleaning but were not in the CSV.
missing = s.index.difference(raw["Date"])
# Compares the row counts, so we can see how many days the CSV lacked.
print(f"Calendar days in range: {len(s)}   Rows in CSV: {len(raw)}   Missing: {len(missing)}")
# Lists those missing dates as readable text like 2004-03-14.
print("Missing dates:", [d.date().isoformat() for d in missing])
# Counts the second kind of gap: the date is there but its temperature is empty.
print(f"Rows present but with an empty value (NaN): {raw['Temp'].isna().sum()}")
# Explains how clean() handles gaps, if there are any.
print("-> Any gap would be filled with a straight line between its neighbours (linear interpolation).")
# Proves the cleaning worked: this number should be 0.
print(f"Empty values left after cleaning: {s.isna().sum()}")

# ----- Question 3: is there a trend or a yearly cycle? -----
banner("3. The whole series: a clear yearly cycle (summer warm, winter cold), no big trend")
# Draws all 10 years as one line, so the yearly wave is visible.
plot_lines({"min temp °C": s}, "Stockholm daily minimum temperature", "year", "°C", dates=s.index)

# Averages all the Januaries, all the Februaries and so on, giving 12 values.
monthly = s.groupby(s.index.month).mean()
# Tells the reader what to expect in the chart below.
print("Average per month (Stockholm: Jan-Feb are the coldest, Jul is the warmest):")
# Draws the 12 monthly averages as bars, with the month names as labels...
plot_bars(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          # ...the values rounded to 1 decimal, and the chart title.
          monthly.round(1), "Average minimum temperature by month (°C)")
# Prints the exact numbers too; :+.1f always shows the sign, like +2.5 or -4.0.
print("Values:", "  ".join(f"{m}: {v:+.1f}" for m, v in monthly.round(1).items()))
# Helps with reading the chart, since negative bars point the other way.
print("(Bars start at 0: below-zero months grow to the LEFT.)")

# ----- Question 4: does the past predict today? -----
# Autocorrelation = how closely a series matches an earlier copy of itself.
banner("4. Autocorrelation: how similar is today to N days ago? (1.0 = identical, 0 = unrelated)")
# The gaps to test: recent days, plus about a quarter, half a year and a full year.
lags = [1, 2, 3, 7, 14, 30, 91, 182, 365]
# Calculates one correlation number for each gap.
corr = [s.autocorr(lag) for lag in lags]
# zip walks through both lists together, pairing each gap with its result.
for lag, c in zip(lags, corr):
    # Prints one aligned line per gap.
    print(f"  today vs {lag:>3} days ago: {c:+.2f}")
# Draws the same numbers as bars, so the pattern is easy to see.
plot_bars([f"{l} days" for l in lags], [round(c, 2) for c in corr], "Correlation of today with N days ago")
# Points out the main finding; corr[0] is the 1-day gap.
print(f"Yesterday is the strongest clue ({corr[0]:+.2f}). Half a year ago is NEGATIVE (opposite season),")
# Connects the finding to the next step: it tells us which features to build.
print("one year ago is positive again (same season). -> We'll build 'lag' and 'season' features.")
