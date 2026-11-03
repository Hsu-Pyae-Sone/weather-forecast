"""
STEP 6 - Improve the model, the honest way: every CHOICE is made on the validation set.
The test set is only used at the very end to report the result.

  1. Hyperparameter search  - try many settings, keep the best on validation
  2. Try a different booster - HistGradientBoosting (the fast, LightGBM-style version)
  3. Retrain on train + validation (more and more recent data), then evaluate on test
Run:  python step6_improve.py   (after step4_train.py)
"""
# itertools, for generating every combination of the settings to try.
import itertools

# joblib loads the step 4 model and saves the improved one.
import joblib
# NumPy, for finding the position of the lowest error.
import numpy as np
# pandas, for the results table and for joining the train and validation data.
import pandas as pd
# The two model types we compare.
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
# Mean Absolute Error, the score used to compare the options.
from sklearn.metrics import mean_absolute_error

# The shared helpers: the model's location, headings, data loading and scoring.
from common import MODEL_PATH, banner, load_splits, print_scores, scores, xy

# Loads all three parts of the data.
train, val, test = load_splits()
# Separates the training data into inputs and answers.
X_train, y_train = xy(train)
# Does the same for the validation data.
X_val, y_val = xy(val)
# Does the same for the test data.
X_test, y_test = xy(test)


# Finds the best tree count for any trained model. Reused so every option is judged the same way.
def best_n_on_val(model):
    """Validation MAE after each tree -> (best number of trees, its MAE)."""
    # Scores the model on the validation days after each tree.
    curve = [mean_absolute_error(y_val, p) for p in model.staged_predict(X_val)]
    # Returns the best tree count (+ 1 because positions start at 0) and its score.
    return int(np.argmin(curve)) + 1, min(curve)


# ----- Part 1: try many combinations of settings -----
banner("1. Grid search: 3 learning rates x 3 depths x 2 subsample values = 18 models")
# The values to try for each setting. Grid search = testing every combination of them.
grid = {"learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]}
# Collects one result row per combination.
rows = []
# itertools.product gives all 18 combinations, like three nested for loops in one line.
for lr, depth, sub in itertools.product(*grid.values()):
    # Builds a model with this combination, allowing up to 600 trees...
    m = GradientBoostingRegressor(n_estimators=600, learning_rate=lr, max_depth=depth,
                                  # ...fixes the random seed so results repeat, then trains it.
                                  subsample=sub, random_state=42).fit(X_train, y_train)
    # Finds this combination's best tree count and its validation score.
    n, mae = best_n_on_val(m)
    # Records the settings used...
    rows.append({"learning_rate": lr, "max_depth": depth, "subsample": sub,
                 # ...and the result they achieved.
                 "n_estimators": n, "val_MAE": round(mae, 3)})
# Turns the rows into a table, with the lowest error (the best) first.
results = pd.DataFrame(rows).sort_values("val_MAE")
# Prints the whole table without row numbers.
print(results.to_string(index=False))
# Picks the top row, which is the winner.
best = results.iloc[0]
# Copies the winning settings into a dictionary, converting to plain Python number types...
best_params = dict(learning_rate=float(best.learning_rate), max_depth=int(best.max_depth),
                   # ...because the model needs real whole numbers for settings like n_estimators.
                   subsample=float(best.subsample), n_estimators=int(best.n_estimators))
# Shows the winning settings.
print(f"\nBest on validation: {best_params}")
# Shows the gap between the best and worst scores...
print(f"Best vs worst of all 18: only {results.val_MAE.max() - best.val_MAE:.3f} °C apart -> "
      # ...and what that small gap tells us.
      "many settings are 'good enough'.")
# Points out a pattern worth looking for in the table above.
print("Note the trade-off: a smaller learning rate needs MORE trees for the same result.")

# ----- Part 2: try a faster type of model -----
banner("2. A different gradient booster: HistGradientBoostingRegressor")
# Same idea, but it first buckets every feature into <=255 bins ("histograms"), so finding a
# split is much faster. This is the trick behind LightGBM; great for big datasets.
# Builds this model with up to 600 trees (it calls that setting max_iter)...
hgb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_depth=3,
                                    # ...turns off random-row early stopping (see step 4), then trains it.
                                    early_stopping=False, random_state=42).fit(X_train, y_train)
# Judges it the same way as the other models.
hgb_n, hgb_mae = best_n_on_val(hgb)
# Prints this model's result...
print(f"HistGradientBoosting: best {hgb_n} trees, validation MAE {hgb_mae:.3f}")
# ...right above the grid search winner's, so they are easy to compare.
print(f"Tuned GradientBoosting:  best {best_params['n_estimators']} trees, validation MAE {best.val_MAE:.3f}")

# ----- Part 3: train the final model and test it -----
banner("3. Final model: best settings, retrained on train + validation (2000-2007)")
# Validation has done its job (choosing settings). Now its 2 years of RECENT data help training.
# Joins the two periods into one larger training set.
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
# Trains the final model with the winning settings; **best_params unpacks the dictionary.
tuned = GradientBoostingRegressor(**best_params, random_state=42).fit(X_trval, y_trval)
# Loads the step 4 model so the two can be compared.
step4_model = joblib.load(MODEL_PATH)
# Makes clear these scores were only reported, never used to choose anything.
print("Test set 2008-2009 (reported once, no decisions were made with it):")
# Scores the simplest baseline...
print_scores({"baseline: tomorrow = today": scores(y_test, test["lag_1"]),
              # ...the model from step 4...
              "step 4 model": scores(y_test, step4_model.predict(X_test)),
              # ...and the new tuned model, all on the same test days.
              "step 6 tuned + more data": scores(y_test, tuned.predict(X_test))})
# Saves the tuned model in the same folder under a new name, keeping the step 4 file intact.
joblib.dump(tuned, MODEL_PATH.with_name("gb_model_tuned.joblib"))
# Calculates the improvement: the step 4 model's error...
gain = (mean_absolute_error(y_test, step4_model.predict(X_test))
        # ...minus the tuned model's error, so a positive number means the tuned model is better.
        - mean_absolute_error(y_test, tuned.predict(X_test)))
# Reports the improvement.
print(f"\nChange in test MAE from tuning + more data: {gain:+.3f} °C (positive = better).")
# Warns against reading too much into such a small difference...
print("A win on validation does not always carry over to the test years: differences this small")
# ...and gives the main lesson of this step...
print("are mostly noise. Lesson: once the settings are reasonable, the model is limited by the")
# ...which is that better inputs now matter more than better settings.
print("INFORMATION in its features, not by its settings.")

# ----- Part 4: ideas for going further -----
banner("4. More ways to improve (try them as exercises!)")
# Prints a list of next ideas; the program does nothing else after this.
print("""- NEW information (the real win here): the GitHub file has more Stockholm columns. Adding
  yesterday's mean and max temperature, cloud cover, pressure, rain and sun cut test MAE from
  1.63 to 1.32 °C in a one-off experiment. (More features built only from past MINIMUM
  temperatures, like lag_365 or a 30-day mean, moved MAE by only ~0.02 °C.)
- Walk-forward validation (sklearn TimeSeriesSplit): validate on several time windows, not one.
- Other libraries: XGBoost, LightGBM, CatBoost - same math, faster, more regularisation options.
- Prediction intervals: loss='quantile' (alpha=0.1 and 0.9) gives a '80% likely range'.
- Remember the ceiling: weather has real randomness; no model reaches MAE 0 from past temps only.""")
