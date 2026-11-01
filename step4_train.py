"""
STEP 4 - Train the real model. The VALIDATION set decides how many trees to keep.

Too few trees  -> underfitting (model too simple, misses the pattern)
Too many trees -> overfitting  (model memorises training noise, gets worse on new days)
We let the validation set (2021-22) tell us where the sweet spot is.
Run:  python step4_train.py
"""
# joblib saves a trained model to a file, so later steps can load it without retraining.
import joblib
# NumPy, used here to find the position of the lowest error.
import numpy as np
# scikit-learn's gradient boosting, the real model for this project.
from sklearn.ensemble import GradientBoostingRegressor
# Mean Squared Error, the formula used to score the model after each tree.
from sklearn.metrics import mean_squared_error

# The shared helpers: the save location, data loading, charts and scoring.
from common import MODEL_PATH, banner, load_splits, plot_lines, print_scores, scores, xy

train, val, _ = load_splits()  # the test set stays locked away until step 5
# Separates the training data into inputs and answers.
X_train, y_train = xy(train)
# Does the same for the validation data.
X_val, y_val = xy(val)

# Hyperparameters = settings WE choose (the model does not learn them).
# Kept in one dictionary, so both models below are guaranteed to use the same settings.
PARAMS = dict(
    learning_rate=0.05,  # how big each correction step is (smaller = slower but safer)
    max_depth=3,         # each tree asks at most 3 yes/no questions
    subsample=0.8,       # each tree trains on a random 80% of rows (adds healthy randomness)
    random_state=42,     # fixed seed -> same result every run
)

# ----- Part 1: train with more trees than we expect to need -----
banner("1. Fit with MANY trees (500) and watch the training log (verbose=1)")
# Explains the columns of the training log that appears below...
print("Iter = tree number | Train Loss = error on training rows | OOB Improve = gain on the")
# ...and defines OOB ("out-of-bag"), the rows a tree did not train on.
print("20% of rows that tree did NOT see ('out-of-bag') | Remaining Time = estimate\n")
# Trains 500 trees and prints progress; **PARAMS unpacks the settings, like JS's {...PARAMS}.
big = GradientBoostingRegressor(n_estimators=500, verbose=1, **PARAMS).fit(X_train, y_train)

# ----- Part 2: find where the model stops improving -----
banner("2. Validation curve: after every tree, how wrong are we on 2021-22?")
# Note: sklearn's built-in early stopping (n_iter_no_change) holds out RANDOM rows, which would
# mix future days into training. For time series we use our chronological validation set instead.
# Training error after each tree; staged_predict gives the predictions at every stage.
train_mse = [mean_squared_error(y_train, p) for p in big.staged_predict(X_train)]
# Validation error after each tree. This is the curve we use to choose the tree count.
val_mse = [mean_squared_error(y_val, p) for p in big.staged_predict(X_val)]
# Finds the position of the lowest error; + 1 turns a position from 0 into a count of trees.
best_n = int(np.argmin(val_mse)) + 1
# Draws both error curves on one chart...
plot_lines({"train MSE": train_mse, "validation MSE": val_mse},
           # ...with this title...
           "Learning curve: training keeps improving, validation stops improving",
           # ...and these axis labels.
           "number of trees", "MSE")
# Reports the best error and how many trees it took...
print(f"Lowest validation MSE {min(val_mse):.3f} at {best_n} trees "
      # ...and compares it with using all 500 trees (val_mse[-1] is the last value in the list).
      f"(with all 500 trees: {val_mse[-1]:.3f}).")
# Explains how to read the chart in terms of underfitting and overfitting...
print("Left of that point = underfitting. Right of it = overfitting (train keeps dropping,")
# ...and why we keep that tree count.
print("validation does not). We keep the sweet spot.")

# ----- Part 3: train the final model and save it -----
banner(f"3. Re-train with the best number of trees ({best_n}) and save the model")
# Trains a fresh model with the chosen tree count and the same settings.
model = GradientBoostingRegressor(n_estimators=best_n, **PARAMS).fit(X_train, y_train)
# Scores the model on data it has seen...
print_scores({"train (seen)": scores(y_train, model.predict(X_train)),
              # ...and on data it hasn't, because comparing the two reveals overfitting.
              "validation (unseen)": scores(y_val, model.predict(X_val))})
# Explains how to read the gap between those two rows.
print("Train is a bit better than validation - normal. A HUGE gap would mean overfitting.")

# Creates the models/ folder if it doesn't exist yet.
MODEL_PATH.parent.mkdir(exist_ok=True)
# Saves the trained model to disk, ready for steps 5 and 6.
joblib.dump(model, MODEL_PATH)
# Confirms where it was saved, showing the path relative to the project folder.
print(f"\nSaved -> {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")
