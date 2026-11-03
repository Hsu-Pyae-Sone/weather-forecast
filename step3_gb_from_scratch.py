"""
STEP 3 - Gradient boosting from scratch, so the math is not a black box.

The whole algorithm (squared-error loss):
    F0(x) = mean(y)                                  start with the average
    repeat M times:
        r  = y - F(x)                                residuals = what is still wrong
                                                     (= the NEGATIVE GRADIENT of the loss)
        h  = small decision tree fitted to r         learn to predict the mistakes
        F(x) = F(x) + learning_rate * h(x)           take a small step to fix them
Run:  python step3_gb_from_scratch.py
"""
# NumPy, for the array math.
import numpy as np
# scikit-learn's own gradient boosting, used in part C to check our version against it.
from sklearn.ensemble import GradientBoostingRegressor
# A ready-made decision tree. We write the boosting loop ourselves but reuse this for each tree.
from sklearn.tree import DecisionTreeRegressor

# The shared helpers: headings, data loading, charts and scoring.
from common import banner, load_splits, plot_lines, print_scores, scores, xy


# Our own version of the algorithm, so the math can be read directly in code.
class SimpleGradientBoosting:
    """Gradient boosting regressor with squared-error loss. Decision trees are the 'weak learners'."""

    # The constructor: stores the settings. No learning happens here.
    def __init__(self, n_trees=200, learning_rate=0.1, max_depth=3):
        # Saves how many trees to build, how big each correction is, and how deep each tree can go.
        self.n_trees, self.learning_rate, self.max_depth = n_trees, learning_rate, max_depth

    # Training: X = the inputs, y = the correct answers.
    def fit(self, X, y):
        self.f0 = y.mean()                              # step 0: best constant guess
        F = np.full(len(y), self.f0)                    # current prediction for every row
        # Stores each trained tree, since predicting later needs all of them.
        self.trees = []
        # The main boosting loop; each pass adds one tree. _ means the counter itself isn't used.
        for _ in range(self.n_trees):
            residuals = y - F                           # step 1: negative gradient of ½(y-F)²
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residuals)  # step 2
            F += self.learning_rate * tree.predict(X)   # step 3: small step towards the answer
            # Remembers this tree for predicting later.
            self.trees.append(tree)
        # Returning self allows chaining, as in Model().fit(X, y).predict(X).
        return self

    # Lets us see how the predictions improve as each tree is added.
    def staged_predict(self, X):
        """Yield the prediction after 1 tree, 2 trees, ... (used to draw learning curves)."""
        # Starts every row at the average saved during training.
        F = np.full(len(X), self.f0)
        # Adds the trees one at a time, in the order they were trained.
        for tree in self.trees:
            # Applies this tree's small correction.
            F = F + self.learning_rate * tree.predict(X)
            # yield hands back the current result, then continues from here on the next request.
            yield F

    # The final prediction, using all the trees at once.
    def predict(self, X):
        # The average, plus every tree's correction scaled down by the learning rate.
        return self.f0 + self.learning_rate * sum(tree.predict(X) for tree in self.trees)


# ----------------------------------------------------------------------------- Part A
banner("A. Toy example you can check with a calculator: 6 days, 1 feature, 3 rounds")
x = np.array([[8], [10], [12], [16], [18], [20]])   # feature: yesterday's temperature
y = np.array([9, 10, 11, 17, 19, 18], dtype=float)  # target:  today's temperature
# The toy example's learning rate: each round fixes half of what the tree suggests.
lr = 0.5

# Round 0: predicts the average for all 6 days.
F = np.full(len(y), y.mean())
# Shows that starting guess.
print(f"Round 0: predict the mean for everyone -> F0 = {y.mean():.1f}")
# Shows the starting loss. Loss = the error score we want to shrink; MSE = mean squared error.
print(f"         loss (MSE) = {np.mean((y - F) ** 2):.2f}")
# Runs rounds 1, 2 and 3.
for m in range(1, 4):
    r = y - F                                                       # residuals
    stump = DecisionTreeRegressor(max_depth=1).fit(x, r)            # a 1-split tree ("stump")
    # Reads the value the tree chose to split on from its top node (node 0).
    thr = stump.tree_.threshold[0]
    # Reads the prediction stored in each of the two leaves (node 1 = left, node 2 = right).
    left, right = stump.tree_.value[1, 0, 0], stump.tree_.value[2, 0, 0]
    # Updates the predictions using a half step of the stump's correction.
    F = F + lr * stump.predict(x)
    # Shows what is still wrong at the start of this round.
    print(f"\nRound {m}: residuals r = y - F = {r.round(2).tolist()}")
    # Shows the rule this round's stump learned...
    print(f"         stump rule: if yesterday <= {thr:.0f}: {left:+.2f}  else: {right:+.2f}"
          # ...and where its two numbers come from.
          "   (each leaf = average residual in it)")
    # Shows the updated predictions after the correction.
    print(f"         F = F + {lr} * stump  ->  {F.round(2).tolist()}")
    # Shows the new loss, which should be lower than last round's.
    print(f"         loss (MSE) = {np.mean((y - F) ** 2):.2f}")
# Shows the true answers, so they can be compared with the final predictions above.
print(f"\nTrue y = {y.tolist()}  -> every round the loss goes down; that's 'boosting'.")
# Explains how the tree picks its split.
print("The split is chosen where it reduces the squared error of the residuals the most.")

# ----------------------------------------------------------------------------- Part B
banner("B. The same algorithm on the real temperature data")
train, val, _ = load_splits()  # test set is NOT used in this step
# Separates the training inputs and answers, converting them from pandas to plain NumPy arrays.
X_train, y_train = (a.to_numpy() for a in xy(train))
# Does the same for the validation data.
X_val, y_val = (a.to_numpy() for a in xy(val))

# Creates our model and trains it on 2000-2005.
ours = SimpleGradientBoosting(n_trees=200, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
# The training error after 1 tree, 2 trees, ... 200 trees.
train_curve = [np.mean((y_train - p) ** 2) for p in ours.staged_predict(X_train)]
# The same on validation days, which the model never trained on. This is the honest score.
val_curve = [np.mean((y_val - p) ** 2) for p in ours.staged_predict(X_val)]
# Picks a few checkpoints to print, since 200 lines would be too many.
for n in (1, 5, 10, 25, 50, 100, 200):
    # Prints both errors at that checkpoint; n - 1 because list positions start at 0.
    print(f"  after {n:>3} trees: train MSE {train_curve[n - 1]:6.2f}   validation MSE {val_curve[n - 1]:6.2f}")
# Draws both error curves on one chart...
plot_lines({"train MSE": train_curve, "validation MSE": val_curve},
           # ...with this title and these axis labels.
           "Loss vs number of trees (our from-scratch model)", "number of trees", "MSE")

# ----- Part C: did we get it right? Compare with the professional library -----
banner("C. Sanity check: scikit-learn's GradientBoostingRegressor does the same thing")
# Creates scikit-learn's version with the same settings as ours...
sk = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3,
                               # ...fixes its random seed so results repeat, then trains it.
                               random_state=0).fit(X_train, y_train)
# Prints both models' validation scores: ours on this line...
print_scores({"our SimpleGradientBoosting": scores(y_val, ours.predict(X_val)),
              # ...and scikit-learn's on this one. Close numbers mean our version is correct.
              "sklearn GradientBoosting": scores(y_val, sk.predict(X_val))})
# Explains why the two scores may not match exactly.
print("(validation set; small differences only come from tie-breaking inside the trees)")
