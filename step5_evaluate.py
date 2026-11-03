"""
STEP 5 - Final exam: evaluate ONCE on the test set (2008-2009), which no step has touched yet.

A score means nothing alone, so we compare against "dumb" baselines.
If a fancy model can't beat "tomorrow = today", it isn't worth shipping.
Run:  python step5_evaluate.py   (after step4_train.py)
"""
# joblib loads the model that step 4 saved.
import joblib
# pandas, for keeping predictions and importances in labelled columns.
import pandas as pd

# The shared helpers: feature names, the model's location, data loading and printing...
from common import (FEATURES, MODEL_PATH, banner, load_splits, plot_bars, plot_hist, plot_lines,
                    # ...plus the scoring functions and the X/y separator.
                    print_scores, scores, xy)

# Gets train (only for one baseline) and test. The validation part isn't needed here.
train, _, test = load_splits()
# Separates the test data into inputs and answers.
X_test, y_test = xy(test)
# Loads the trained model from disk, so there's no need to train again.
model = joblib.load(MODEL_PATH)
# Predicts every test day and labels each prediction with its date.
pred = pd.Series(model.predict(X_test), index=y_test.index)

# ----- Part 1: is the model better than simple guesses? -----
banner(f"1. Scores on {len(y_test)} unseen days vs simple baselines")
# Prints one row per approach. A baseline is a simple guess that sets the score to beat.
print_scores({
    # Baseline: always guess the average temperature of the training years.
    "baseline: average of train": scores(y_test, [train["target"].mean()] * len(y_test)),
    "baseline: tomorrow = today": scores(y_test, test["lag_1"]),         # 'persistence'
    # Baseline: guess the average of the previous 7 days.
    "baseline: last 7-day mean": scores(y_test, test["roll_mean_7"]),
    # Our actual model.
    "GRADIENT BOOSTING": scores(y_test, pred),
})
# Prints a plain-language reminder of what each score means.
print("""
MAE  = average size of the miss in °C            ("on average the forecast is off by this much")
RMSE = like MAE but punishes big misses more     (square -> average -> square root)
R²   = share of the day-to-day variation we explain (1 = perfect, 0 = as good as the average)""")

# ----- Part 2: see the predictions next to the real values -----
banner("2. Actual vs predicted - last 120 days of the test set")
# A reusable "last 120 rows" selector, the same as writing [-120:].
last = slice(-120, None)
# Draws the real and predicted temperatures for those days as two lines...
plot_lines({"actual": y_test.iloc[last], "predicted": pred.iloc[last]},
           # ...with this title and these axis labels.
           "Actual vs predicted (Sep-Dec 2009)", "day", "°C")
# Describes what the chart shows...
print("The model follows the season and the general level well, but smooths out sudden")
# ...and why: the inputs contain no warning that a sudden change is coming.
print("jumps - a cold front tomorrow is simply not visible in the past days' temperatures.")

# ----- Part 3: are the errors balanced? -----
banner("3. Residuals (actual - predicted): are the errors balanced around 0?")
# The error for each day: positive means the model guessed too low, negative too high.
residuals = y_test - pred
# Shows how the errors are spread; a hill centred on 0 is what we want.
plot_hist(residuals, "Histogram of errors (°C)")
# Reports the average error, which reveals whether the model leans too high or too low...
print(f"Mean error {residuals.mean():+.2f} °C (close to 0 = no systematic bias), "
      # ...then the share of days within 2 °C, and the single worst miss.
      f"{(residuals.abs() <= 2).mean():.0%} of days within ±2 °C, worst miss {residuals.abs().max():.1f} °C.")

# ----- Part 4: which months are hardest? -----
banner("4. Where does it struggle? Average error by month")
# Takes the size of each error and averages it per calendar month.
month_mae = residuals.abs().groupby(residuals.index.month).mean()
# Draws those 12 averages as bars, with the month names as labels...
plot_bars(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          # ...the values rounded to 2 decimals, and the chart title.
          month_mae.round(2), "MAE by month (°C)")

# ----- Part 5: which inputs mattered most? -----
banner("5. Feature importance: which inputs did the trees use most?")
# Reads the importance scores the model calculated, names them, and sorts biggest first.
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
# Draws them as bars, with the most useful input at the top.
plot_bars(imp.index, imp.round(3), "Share of total error reduction per feature")
# Explains what an importance score measures...
print("Importance = how much each feature's splits reduced the squared error, summed over all")
# ...and that all the scores together add up to 1.
print("trees and scaled to add up to 1.")
