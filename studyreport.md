# Study Report — Week 2: Gradient Boosting for Time-Series Forecasting

| | |
|---|---|
| **Course** | AI/ML Class — Week 2 assignment |
| **Topic** | Classical machine learning: Gradient Boosting (decision-tree ensembles) |
| **Problem** | Predict **tomorrow's minimum temperature in Stockholm, Sweden** from past days (time-series regression) |
| **Dataset** | [Weather prediction dataset](https://github.com/florian-huber/weather_prediction_dataset) on GitHub (Stockholm column, 2000–2009), built from ECA&D station data |
| **Tools** | Python 3.11, pandas, NumPy, scikit-learn, plotext (charts in the terminal), matplotlib (the same charts saved as PNGs) |
| **Result** | Test MAE **1.63 °C**, compared with **1.68 °C** for the naive baseline (≈ 3% better; RMSE ≈ 5% better), R² = 0.89 |
| **Date** | 3 Nov 2026 |

---

## Table of contents
1. [The big picture (end-to-end)](#1-the-big-picture-end-to-end)
2. [The dataset](#2-the-dataset)
3. [Data processing & transformation, step by step](#3-data-processing--transformation-step-by-step)
4. [The model: gradient boosting — basics](#4-the-model-gradient-boosting--basics)
5. [The mathematics of gradient boosting](#5-the-mathematics-of-gradient-boosting)
6. [Training with a validation set](#6-training-with-a-validation-set)
7. [Model evaluation](#7-model-evaluation)
8. [How to improve the model](#8-how-to-improve-the-model)
9. [Evolution, current state, open problems](#9-evolution-current-state-open-problems)
10. [Why this matters for my roles (SWE / frontend / UI-UX)](#10-why-this-matters-for-my-roles)
11. [Glossary](#11-glossary)
12. [Check my understanding](#12-check-my-understanding)
13. [References](#13-references)

---

## 1. The big picture (end-to-end)

```
 ┌──────────┐   ┌──────────┐   ┌────────────────┐   ┌───────────────┐   ┌─────────────┐   ┌───────────┐
 │ 0. FETCH │──►│ 1. CLEAN │──►│ 2. FEATURES    │──►│ 3. SPLIT      │──►│ 4. TRAIN    │──►│ 5. EVAL   │
 │ GitHub   │   │ check:   │   │ lags, rolling  │   │ by TIME:      │   │ gradient    │   │ on test,  │
 │ → CSV    │   │ no gaps  │   │ mean, season   │   │ train│val│test│   │ boosting    │   │ vs        │
 │ 3,653    │   │ found    │   │ (8 columns)    │   │ 60% │20%│20% │   │ val picks   │   │ baselines │
 │ days     │   │ → 3,653  │   │ → 3,646 rows   │   │               │   │ # of trees  │   │           │
 └──────────┘   └──────────┘   └────────────────┘   └───────────────┘   └─────────────┘   └───────────┘
   common.py      step1            step2                 step2              step3, step4      step5, step6
```

| Step | Script | Input → Output | Key idea |
|---|---|---|---|
| 0 | `common.download()` | GitHub CSV → smaller local CSV | Data comes from a public URL, the same way a frontend `fetch()`es JSON |
| 1 | `step1_explore.py` | CSV → understanding | Look before you model: gaps, cycles, correlations |
| 2 | `step2_features.py` | 1 column → 8-feature table, 3 splits | ML needs `X → y` rows; never let the future leak in |
| 3 | `step3_gb_from_scratch.py` | table → our own model | Boosting = repeatedly fit a small tree to the current errors |
| 4 | `step4_train.py` | train + val → saved model | Validation tells us when to stop adding trees |
| 5 | `step5_evaluate.py` | model + test → scores | Compare with "dumb" baselines; study the errors |
| 6 | `step6_improve.py` | grid of settings → tuned model | Tune only on validation; more information beats more tuning |

> **Web-dev analogy:** This pipeline is like a build pipeline. Raw assets go through lint/clean, transform/bundle, test, and then ship. `models/gb_model.joblib` is the build artifact.

---

## 2. The dataset

**Daily minimum temperature, Stockholm, Sweden, 2000–2009**
- **Source:** the [Weather prediction dataset](https://github.com/florian-huber/weather_prediction_dataset) on GitHub, made by Huber et al. for teaching machine learning. It covers 18 European locations and is built from [ECA&D](https://www.ecad.eu) (European Climate Assessment & Dataset) station records.
- **Licence:** **CC BY 4.0** for the [dataset on Zenodo](https://doi.org/10.5281/zenodo.7525955) and **MIT** for the GitHub repo. It is free to use, but you must credit the source (see [References](#13-references)).
- **How we get it:** `common.download()` reads the CSV straight from GitHub (no key needed), keeps only the date and `STOCKHOLM_temp_min` columns, and saves a simple `Date,Temp` CSV to `data/`. It is downloaded only on the first run.
- **Shape:** the source file has 3,654 rows × 165 columns. We keep 3,653 rows × 2 columns, dropping the file's last day (2010-01-01) so we have exactly 10 full years. One row is one day; `Temp` is that day's **minimum** temperature in °C.

```
GET https://raw.githubusercontent.com/florian-huber/weather_prediction_dataset/main/dataset/weather_prediction_dataset.csv
  ⇒ DATE=20000101, STOCKHOLM_temp_min=-9.3, … (163 other columns we ignore)
  ⇒ CSV row:  2000-01-01,-9.3
```

| Statistic | Value |
|---|---|
| Date range | 2000-01-01 → 2009-12-31 |
| Mean | 5.11 °C |
| Std (typical spread around the mean) | 7.25 °C |
| Min / Median / Max | **−19.7** / 5.0 / 21.2 °C |
| Missing days | **0** |

**Average minimum temperature by month (°C)**

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| −2.6 | **−3.2** | −1.6 | +2.8 | +7.4 | +11.6 | **+14.9** | +14.5 | +10.2 | +5.6 | +2.0 | −0.9 |

**Why it is a good beginner dataset**
- Only one variable, so we can focus on *time-series thinking* instead of data wrangling.
- Very strong **seasonality** (a pattern that repeats every year): about 18 °C between the February and July averages.
- **No clear trend** across the 10 years. Yearly averages bounce between 4.7 and 5.8 °C.
- Real **noise**, especially in winter, when the minimum can jump by 10 °C overnight. That keeps our expectations honest.

```
 °C   Stockholm daily minimum temperature (sketch of the step-1 terminal chart)
 20 ┤  ▐█▌    ▐█▌    ▐█▌    ▐█▌    ▐█▌    ▐█▌    ▐█▌    ▐█▌    ▐█▌    ▐█▌      ← summers
 10 ┤ ▐███▌  ▐███▌  ▐███▌  ▐███▌  ▐███▌  ▐███▌  ▐███▌  ▐███▌  ▐███▌  ▐███▌
  0 ┤█████████████████████████████████████████████████████████████████████
−10 ┤▐▌    ▐█▌    ▐▌     ▐█▌    ▐▌     ▐▌     ▐█▌    ▐█▌    ▐█▌    ▐█▌        ← winters
−20 ┤        ▘  (coldest night: −19.7 °C, in 2001)
    └─2000───2001───2002───2003───2004───2005───2006───2007───2008───2009
```

### Autocorrelation: does the past predict the future?
**Autocorrelation** is the correlation between a series and a shifted copy of itself. It answers: "if it was cold N days ago, is it cold today?" A value of +1 means the two move perfectly together, 0 means they are unrelated, and −1 means they move in opposite directions.

| Today vs… | 1 day ago | 2 | 3 | 7 | 14 | 30 | 91 | 182 | 365 |
|---|---|---|---|---|---|---|---|---|---|
| correlation | **+0.95** | +0.90 | +0.88 | +0.82 | +0.77 | +0.69 | −0.01 | **−0.76** | **+0.79** |

- **Yesterday is a very strong clue (0.95).** That's why we build "lag" features. It also means the simple rule "tomorrow = today" is a *tough* baseline here.
- Half a year ago the correlation is **negative** because it was the opposite season. One year ago it is positive again because it was the same season. So we also give the model **calendar features**.

---

## 3. Data processing & transformation, step by step

### Step 3.1: Clean the gaps (`common.clean`)
This dataset has **no gaps**, so this step changes nothing here. It stays in the pipeline as a safety net, because real data usually has holes. Here is what it would do (the numbers are made up):
```
raw:    ... 05-30       06-01 ...              (May 31 is missing)
asfreq: ... 05-30       05-31=NaN  06-01       (insert every calendar day)
interp: ... 14.5        13.2       11.9        (14.5+11.9)/2 = 13.2, straight line between neighbours
```
- `asfreq("D")` forces exactly **one row per day**. Many time-series tools assume evenly spaced steps, and a silent gap would make `lag_1` mean "2 days ago" for one row.
- `interpolate()` fills each hole with a straight line between its neighbours. That is reasonable for a day or two, but not for a long gap.
- Step 1 also checks for empty (`NaN`) values; there were none. The dataset's authors already replaced invalid readings with the column's average, but Stockholm's values show no sign of this: all have exactly one decimal, which an average would be unlikely to.

### Step 3.2: The sliding window, from a series to a supervised table
A model like gradient boosting learns from rows of **inputs (X) → answer (y)**. A time series only has one column, so we **create** the inputs from the past with `shift()`:

```
 date        temp            target(y)  lag_1  lag_2  lag_3
 01-01       -9.3     ──►      -9.3      NaN    NaN    NaN
 01-02        0.5               0.5     -9.3    NaN    NaN
 01-03       -1.0              -1.0      0.5   -9.3    NaN
 01-04        2.5               2.5     -1.0    0.5   -9.3   ← first complete row
 01-05       -1.8              -1.8      2.5   -1.0    0.5
```
Each row now says: *"given the last few days, today was X."* Rows with `NaN` (not enough history yet) are dropped.

> **Frontend analogy:** `lag_1` is like a `usePrevious(value)` hook: the state from the previous render.

### Step 3.3: Rolling mean
`roll_mean_7 = s.shift(1).rolling(7).mean()` is the average of the **7 days before** today. It smooths out noise, much like **debouncing** smooths out noisy input events. The `shift(1)` comes first so that today's own temperature is never part of its own input.

### Step 3.4: Calendar features and cyclic encoding
`month` (1–12) tells the model the season. Plain *day-of-year* has a problem, though: Dec 31 is 365 and Jan 1 is 1, so they look far apart even though they are neighbours. The fix is to place the day on a **circle**:

$$\text{doy\_sin} = \sin\left(\frac{2\pi \cdot \text{doy}}{365.25}\right), \qquad \text{doy\_cos} = \cos\left(\frac{2\pi \cdot \text{doy}}{365.25}\right)$$

```
                Apr 1 (sin=1, cos=0)
                     ●
                 ╱       ╲
   Jul 1 ●     (circle)      ● Jan 1 (0.017, 1.000)
 (0, −1)         ╲       ╱   ● Dec 31 (−0.004, 1.000) ← distance only 0.021
                     ●
                   Oct 1
```
> **UI/UX analogy:** This is exactly the **color hue wheel**. A hue of 359° is almost the same color as 0°. Designers think of hue as an angle on a circle, not a number on a line.

### Step 3.5: The final feature table (8 features + 1 target, 3,646 rows)

| Feature | Meaning | Why it helps |
|---|---|---|
| `lag_1`, `lag_2`, `lag_3` | temperature 1, 2, 3 days ago | weather is "sticky" day to day |
| `lag_7` | temperature 1 week ago | slower trends |
| `roll_mean_7` | average of the previous 7 days | denoised recent level |
| `month` | 1..12 | season |
| `doy_sin`, `doy_cos` | position in the year on a circle | smooth seasonality, no Dec/Jan jump |

**What we did *not* do:** scaling or normalising. Many models (linear regression, neural nets) need features on similar scales. **Trees don't.** A split like `lag_1 ≤ −8.5` works the same if every value is multiplied by 100. Negative numbers are no problem either.

### Step 3.6: Train / validation / test split, by time
```
|---------- train (learn) ----------|--- validation (tune) ---|---- test (final exam) ----|
2000-01-08                 2005-12-31 2006-01-01     2007-12-31 2008-01-01       2009-12-31
          2,185 rows (60%)                  730 rows (20%)              731 rows (20%)
```
- **Train:** the model learns its trees from this data.
- **Validation:** *we* use this to make decisions, such as how many trees and which settings. The model never trains on it while we tune.
- **Test:** locked away until the end. It plays the role of real "future" days.

**Why not a random split?** In a time series, shuffling would put 2009 days into training and 2002 days into the test set. The model would then "remember the future", and the score would look better than real life. This is called **data leakage**.

> **Software analogy:** Train/validation/test is like dev, staging, and production. Or, for a student: practice questions, a mock exam, and the final exam. If you study the final-exam questions, your grade no longer measures anything.

---

## 4. The model: gradient boosting — basics

### 4.1 Building block: the decision tree
A **decision tree** is a flowchart of yes/no questions on the features. Each **leaf** (end point) outputs a number. Here is an illustrative depth-2 tree (the numbers are made up):

```
                  lag_1 ≤ 2.5 ?
                 ╱              ╲
             yes                  no
       lag_1 ≤ −8.5 ?          lag_1 ≤ 10.4 ?
       ╱         ╲              ╱        ╲
   predict      predict      predict    predict
   −13.1 °C     −2.4 °C       6.2 °C    13.9 °C
```
> **Frontend analogy:** A tree is nested `if/else`, or a set of **CSS media-query breakpoints**. The output is constant between breakpoints and jumps at the thresholds. The tree's job during training is to *find the best breakpoints* automatically.

A single deep tree overfits (it memorises noise), and a single shallow tree is too crude. **Ensembles** (many trees combined) fix this.

### 4.2 Two ways to combine trees

| | **Bagging / Random Forest** | **Boosting** |
|---|---|---|
| How trees are built | independently, in parallel, each on a random sample | **one after another**, each fixing the previous ones' mistakes |
| Each tree is | deep (strong but noisy) | shallow (a "weak learner") |
| Combine by | averaging | **adding up** small corrections |
| Analogy | ask 100 people and average their answers | a draft that a series of editors refine, each fixing what is still wrong |

### 4.3 The core intuition of boosting
1. Start with a dumb guess: **the average temperature**.
2. Look at the **errors** (residuals = actual − predicted).
3. Train a small tree to **predict those errors**.
4. Add a **small fraction** (the learning rate) of that tree to the model.
5. Repeat many times. Each new tree focuses on what is *still* wrong.

> **Frontend analogy (this is almost literally the algorithm):** A smooth animation "lerp" loop does
> `x = x + 0.1 * (target - x)` on every frame.
> Gradient boosting does `F = F + learning_rate * tree(target - F)` on every round.
> The only difference is that the model does not know the target for *new* days, so a tree has to *learn* a rule that predicts `target - F` from the features.

### 4.4 Hyperparameters used in this project
A **hyperparameter** is a setting we choose; the model doesn't learn it.

| Hyperparameter | Meaning | Our value | Too small → | Too large → |
|---|---|---|---|---|
| `n_estimators` | number of trees (rounds) | chosen by validation (115 in step 4, 64 in step 6) | underfit | overfit |
| `learning_rate` | step size per tree | 0.05 (step 4), 0.1 (step 6) | needs many trees | unstable, overfits fast |
| `max_depth` | questions per tree | 3 (step 4), 4 (step 6) | can't capture interactions | each tree overfits |
| `subsample` | share of rows each tree sees | 0.8 | noisy | less regularisation |

---

## 5. The mathematics of gradient boosting

### 5.1 Notation
- $x_i$: the features of day $i$ (our 8 numbers); $y_i$: the true temperature.
- $F(x)$: the model's prediction. It is a **sum of trees**:

$$F_M(x) = F_0 + \eta\, h_1(x) + \eta\, h_2(x) + \dots + \eta\, h_M(x)$$

- $h_m$: the $m$-th small tree. $\eta$ (eta): the learning rate. $M$: the number of trees.

### 5.2 The loss function measures how wrong we are
For regression we use the **squared error**. The ½ is there only to make the derivative tidy:

$$L(y, F) = \tfrac{1}{2}\,(y - F)^2 \qquad \text{Total loss} = \sum_i L(y_i, F(x_i))$$

### 5.3 Why it's called *gradient* boosting
The **gradient** is the derivative: the direction in which the loss goes *up*. To reduce the loss, we move in the **negative** gradient direction:

$$\frac{\partial L}{\partial F} = -(y - F) \quad\Longrightarrow\quad -\frac{\partial L}{\partial F} = y - F = \textbf{residual}$$

So with squared loss, **the negative gradient is exactly the residual**. "Fit a tree to the residuals" *is* gradient descent. Instead of nudging a few weights (as in linear regression or neural nets), we nudge the **predictions themselves**, using a tree to decide how to nudge them for any input. This is why the method is described as **gradient descent in function space**.

```
 Normal gradient descent (weights):        Gradient boosting (functions):
   w ← w − η · ∂Loss/∂w                      F ← F + η · h,  where h ≈ −∂Loss/∂F  (a tree)
```

### 5.4 The full algorithm (Friedman, 2001)
```
Input: data (x_i, y_i), loss L, number of trees M, learning rate η, tree depth d

1. F_0 = argmin_c Σ L(y_i, c)                 # best constant  → for squared loss: mean(y)
2. for m = 1 … M:
     a. r_i = −∂L(y_i, F)/∂F  at F = F_{m−1}  # pseudo-residuals → for squared loss: y_i − F(x_i)
     b. fit a regression tree h_m to (x_i, r_i) with depth ≤ d
     c. each leaf's value = the best constant for the rows in it
                                              # → for squared loss: the MEAN residual in that leaf
     d. F_m(x) = F_{m−1}(x) + η · h_m(x)      # small step
3. return F_M
```
Our `SimpleGradientBoosting` class in `step3_gb_from_scratch.py` implements this in about 15 lines.

### 5.5 How a tree chooses a split
For each feature and each possible threshold, the tree splits the rows into *left* and *right* groups and predicts the mean residual on each side. It picks the split with the **smallest sum of squared errors (SSE)**:

$$\text{SSE(split)} = \sum_{i \in \text{left}} (r_i - \bar r_{\text{left}})^2 + \sum_{i \in \text{right}} (r_i - \bar r_{\text{right}})^2$$

A deeper tree repeats this inside each group.

### 5.6 Worked example: you can check it with a calculator (step 3, part A)
Six days, one feature (yesterday's temperature), learning rate $\eta = 0.5$, trees of depth 1 (a **stump**, i.e. a tree with a single question):

| x = yesterday | 8 | 10 | 12 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|
| **y = today** | 9 | 10 | 11 | 17 | 19 | 18 |

**Round 0:** $F_0 = \text{mean}(y) = 84/6 = 14$. Every prediction is 14. MSE $= (25+16+9+9+25+16)/6 = 16.67$.

**Round 1:** residuals $r = y - F = [-5, -4, -3, 3, 5, 4]$. The tree tries every threshold:

| threshold | left residuals | right residuals | SSE |
|---|---|---|---|
| x ≤ 9 | [−5] | [−4,−3,3,5,4] mean 1 | 70.0 |
| x ≤ 11 | [−5,−4] | [−3,3,5,4] | 39.25 |
| **x ≤ 14** | **[−5,−4,−3] mean −4** | **[3,5,4] mean +4** | **4.0** ← best |
| x ≤ 17 | [−5,−4,−3,3] | [5,4] | 39.25 |
| x ≤ 19 | [−5,−4,−3,3,5] | [4] | 80.8 |

Stump: `if yesterday ≤ 14: −4 else +4`. Update: $F = 14 + 0.5 \cdot (\mp 4)$, giving `[12,12,12,16,16,16]`. **MSE = 4.67**

**Round 2:** $r = [-3,-2,-1,1,3,2]$. The same split wins, with leaves ∓2. $F$ = `[11,11,11,17,17,17]`. **MSE = 1.67**

**Round 3:** $r = [-2,-1,0,0,2,1]$. Now a *different* split wins: `yesterday ≤ 11` with leaves −1.5 / +0.75. (`≤ 17` ties at SSE 3.25; the library picks the first.)
$F$ = `[10.25, 10.25, 11.38, 17.38, 17.38, 17.38]`. **MSE = 0.82**

```
MSE: 16.67 ──► 4.67 ──► 1.67 ──► 0.82     every round fixes part of what is still wrong
```
Notice the **learning rate** at work. Each round fixes only **half** of the error the stump found, so many small steps add up. Small steps generalise better than one big jump, because no single tree can push the model too far toward noise.

### 5.7 Two extras used in this project
- **Stochastic gradient boosting** (`subsample=0.8`): each tree sees a random 80% of rows. The randomness acts as **regularisation** (anything that discourages memorising noise). It also gives the **OOB ("out-of-bag") improvement** numbers in the training log, which are measured on the 20% of rows that tree didn't see.
- **Feature importance**: for each feature, add up how much SSE its splits removed across all trees, then scale so the total is 1.

### 5.8 Beyond squared loss (for later)
Change $L$ and the algorithm stays the same. Only the pseudo-residual changes:

| Task | Loss | Pseudo-residual $r_i$ |
|---|---|---|
| Regression | squared: ½(y−F)² | y − F |
| Robust regression | absolute: \|y−F\| | sign(y − F) |
| Binary classification | log-loss, F = log-odds | y − sigmoid(F) |
| Prediction intervals | quantile (pinball) | α or α−1 |

**XGBoost** additionally uses the **second derivative** (a Newton step) and adds penalties on the number of leaves and on the size of leaf values. LightGBM and CatBoost build on the same core idea.

### 5.9 Seven-line summary of the math
1. Prediction = average + many small tree corrections.
2. The loss measures error; its gradient says which way to move each prediction.
3. For squared loss, the negative gradient = the residual.
4. Each tree learns to predict the residual from the features (it picks splits that minimise SSE).
5. Leaf value = the mean residual in that leaf.
6. Add the tree × learning rate. Repeat.
7. Stop when validation error stops improving.

---

## 6. Training with a validation set

### 6.1 The from-scratch model on real data (step 3, part B)
Learning rate 0.1, depth 3:

| trees | 1 | 5 | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| train MSE | 46.43 | 23.43 | 11.75 | 5.12 | 4.30 | 3.79 | **3.06** ↓ keeps falling |
| validation MSE | 45.41 | 22.65 | 11.45 | 5.46 | **5.18** | 5.22 | 5.30 ↑ rising again |

Our 15-line version and scikit-learn's `GradientBoostingRegressor` give practically the same validation result: MAE 1.789 vs 1.789 °C, RMSE 2.303 vs 2.301 °C, R² 0.903 for both. The tiny difference comes from how ties between equally good splits are broken. So the library really is doing what we wrote.

### 6.2 Underfitting vs overfitting (step 4)
Step 4 trains 500 trees (learning rate 0.05, depth 3, subsample 0.8) and measures the validation error after every tree. The sketch below shows the shape of the terminal chart:

```
 MSE
 50 ┤╲
    │ ╲
 20 ┤  ╲___
    │      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  validation (flat ≈ 5.2, then slowly up)
  4 ┤   ↑ ╲____
    │ best=115 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲_____________  train (keeps going down)
  3 ┤
    └───────────────────────────────────────────────── trees → 500
     underfit │ sweet spot │          overfit
```
- **Lowest validation MSE: 5.161 at 115 trees.** With all 500 trees it is 5.387, which is worse. Extra trees just memorise training noise.
- This is **early stopping chosen by validation**. We don't use scikit-learn's built-in `n_iter_no_change`, because it holds out *random* rows, which would leak future days into training.

### 6.3 The saved model (115 trees)

| | MAE °C | RMSE °C | R² |
|---|---|---|---|
| train (seen) | 1.596 | 2.047 | 0.925 |
| validation (unseen) | 1.774 | 2.272 | 0.906 |

Train is better than validation, as expected. The gap (≈ 0.2 °C) is small: the validation R² (0.906) is close to the training R² (0.925).

### 6.4 Reading the `verbose=1` training log
```
 Iter   Train Loss   OOB Improve   Remaining Time
    1      51.7967        4.5066            3.25s
   10      23.9817        3.5959            1.30s
  100       4.3224        0.2102            0.86s
  500       2.6725       -0.1746            0.00s
```
`Train Loss` goes down steadily, which is the model fitting the training data. `OOB Improve` is large but jumpy early on, and stays close to 0 after about 40 trees. That is another hint that later trees add little real knowledge. When there is **no loss left to reduce** on new data, more training stops helping.

---

## 7. Model evaluation

### 7.1 Metrics
With $e_i = y_i - \hat y_i$ (the error on day $i$, where $\hat y_i$ is the prediction):

| Metric | Formula | Plain meaning | Units |
|---|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert e_i\rvert$ | average size of a miss | °C |
| **RMSE** | $\sqrt{\frac{1}{n}\sum e_i^2}$ | like MAE but punishes big misses more | °C |
| **R²** | $1 - \frac{\sum e_i^2}{\sum (y_i - \bar y)^2}$ | share of the variation we explain (1 = perfect, 0 = no better than the average) | — |

### 7.2 Results on the test set (731 unseen days, 2008–2009)
A score means nothing alone, so we compare it with **baselines**: simple rules with no ML.

| Model | MAE °C | RMSE °C | R² |
|---|---|---|---|
| Baseline: always predict the training average | 5.311 | 6.370 | 0.000 |
| Baseline: last 7-day mean | 2.262 | 2.811 | 0.805 |
| Baseline: **tomorrow = today** ("persistence") | 1.680 | 2.178 | 0.883 |
| **Gradient boosting (step 4)** | **1.632** | **2.070** | **0.894** |
| Gradient boosting tuned + more data (step 6) | **1.606** | **2.048** | **0.897** |

**Interpretation**
- The model beats every baseline: about **3% better MAE** and **5% better RMSE** than "tomorrow = today". RMSE improves more because the model mainly reduces the *big* misses, which squared errors punish most.
- In Stockholm, yesterday's temperature is already very informative (autocorrelation 0.95), so the naive rule is strong and the margin for ML is small. In the Melbourne data used in an earlier draft of this project, the same model beat persistence by about 12%, because there yesterday was a weaker clue (0.77). **How much ML helps depends on the data, not only on the model.**
- R² = 0.89 looks high, but most of it comes from the season, which every reasonable method gets. Always compare R² against the baseline's R² (0.883), not against 0.

### 7.3 Error analysis
- **Residuals** (actual − predicted): the mean is **+0.03 °C**, so there is no systematic bias. **38%** of days are within ±1 °C, **68%** within ±2 °C, and **85%** within ±3 °C.
- **Worst miss: 2009-01-09, off by 9.6 °C.** Stockholm's minimum had stayed between −7.8 and −9.7 °C for three days (−9.7 °C on Jan 8), then jumped to +1.0 °C. The model predicted another cold night (−8.6 °C), because nothing in the past minimum temperatures warned of the jump.
- **MAE by month:** the model is worst in **winter** (Feb 1.97, Dec 1.88, Jan 1.83 °C) and best in **summer** (Aug 1.45 °C), about a 1.4× difference. April is also high (1.82 °C). Clear, calm winter nights can get much colder than cloudy, windy ones, and temperature history alone can't tell those apart.

### 7.4 Feature importance (what the trees used)

| Feature | Importance |
|---|---|
| `lag_1` | **0.809** |
| `roll_mean_7` | 0.161 |
| `doy_cos` | 0.014 |
| `doy_sin` | 0.004 |
| `lag_3` | 0.004 |
| `lag_7` | 0.004 |
| `lag_2` | 0.003 |
| `month` | 0.000 |

Yesterday alone accounts for 81% of the error reduction. The weekly average and the season make small corrections. Caution: impurity-based importance can be misleading when features are correlated (as lags are). SHAP values are a more reliable alternative.

---

## 8. How to improve the model

### 8.1 What we tried (step 6), with every decision made on **validation**
1. **Grid search:** 3 learning rates × 3 depths × 2 subsample values = 18 models. Each gets its best tree count from validation.
   - Best: `learning_rate=0.1, max_depth=4, subsample=0.8, n_estimators=64`, validation MAE **1.748** (step 4's settings scored 1.774).
   - The best and worst of the 18 were only **0.029 °C** apart, so many settings are "good enough".
   - Trade-off visible in the table: a smaller learning rate needs many more trees (at depth 3 and subsample 0.8: lr 0.02 → 425 trees, lr 0.1 → 73).
2. **HistGradientBoostingRegressor:** scikit-learn's LightGBM-style booster. It first buckets features into ≤ 255 bins, which makes it much faster on big data. Validation MAE 1.771, about the same.
3. **Retrain on train + validation** (2000–2007) with the best settings, giving the model two more years of *recent* data. On the test years:
   - MAE **1.632 → 1.606**, RMSE 2.070 → 2.048 (both slightly better).
   - The win on validation carried over this time, but a difference of 0.026 °C on one test period is too small to prove much.
4. **More features** (one-off experiments during development, not in the step scripts; they use step 4's settings and were not used to choose anything):

| Features added to step 4's model | Validation MAE °C | Test MAE °C | Test RMSE °C |
|---|---|---|---|
| None (step 4) | 1.774 | 1.632 | 2.070 |
| More past *minimum* temperatures: `lag_365`, 30-day mean, trend (`lag_1 − lag_2`) | 1.767 | 1.610 | 2.062 |
| Yesterday's cloud cover, pressure, rain, sunshine | 1.746 | 1.556 | 2.009 |
| Yesterday's mean and max temperature | 1.516 | 1.375 | 1.746 |
| **All of yesterday's other Stockholm columns** (the two rows above together) | **1.455** | **1.317** | **1.698** |

### 8.2 The lesson
Once the settings are reasonable, **the model is limited by the information in its inputs, not by its settings.** More settings or more features built from the same column barely helped. New columns cut test MAE by about 19%, far more than any tuning did. Yesterday's mean and max temperature helped most, and yesterday's cloud cover, pressure, rain and sun also added something.

### 8.3 Improvement ideas, ranked by expected impact
| Idea | Why | Effort |
|---|---|---|
| **Add new information**: Stockholm's other columns (mean and max temperature, cloud cover, pressure, rain, sunshine) | measured in 8.1: test MAE 1.63 → 1.32 °C. This file has no wind, humidity or snow data for Stockholm | low: same CSV file, more columns |
| **Walk-forward validation** (`TimeSeriesSplit`) | one validation window can be lucky; several windows give a more trustworthy choice | low |
| **Prediction intervals** (`loss="quantile"`, α = 0.1 and 0.9) | shows "likely −8 to −2 °C"; the range should be wider in winter | low |
| **XGBoost / LightGBM / CatBoost** | same math, faster, more regularisation knobs | low |
| **Multi-day forecasts** (recursive or one model per horizon) | real apps show 7 days ahead | medium |
| **Remove the season first** (model the deviation from the yearly average) | trees then only learn the weather "surprise" | medium |

---

## 9. Evolution, current state, open problems

### 9.1 Evolution of the idea
| Year | Milestone | Problem it solved |
|---|---|---|
| 1984 | **CART** decision trees (Breiman, Friedman, Olshen, Stone) | interpretable models for any tabular data |
| 1988–90 | Kearns & Valiant ask *"can many weak learners make a strong one?"*; Schapire proves **yes** | the theory of boosting |
| 1995–97 | **AdaBoost** (Freund & Schapire) | the first practical boosting: re-weight the examples that were wrong |
| 1999–2001 | **Gradient Boosting Machine** (Friedman); AnyBoost (Mason et al.) | boosting = gradient descent, so it works with *any* differentiable loss |
| 2002 | **Stochastic** gradient boosting (Friedman) | row subsampling gives better generalisation |
| 2001 | Random Forests (Breiman) | the parallel ("bagging") alternative |
| 2014–16 | **XGBoost** (Chen & Guestrin) | second-order steps, regularisation, speed; dominated Kaggle |
| 2017 | **LightGBM** (Microsoft), **CatBoost** (Yandex) | histogram splits and leaf-wise growth (speed); native categorical features |
| 2019 | scikit-learn **HistGradientBoosting** | LightGBM-style speed inside scikit-learn |
| 2020 | **M5 forecasting competition** (Walmart sales) | top solutions were largely LightGBM, showing GBDT is strong for time series |

### 9.2 Current state (2026)
- **Gradient-boosted decision trees (GBDTs) are still the default for tabular data** (spreadsheets, database rows). They are fast to train on a CPU, need little preprocessing, and are hard to beat.
- Common libraries: **scikit-learn** (learning, small data), **XGBoost**, **LightGBM**, **CatBoost** (production, big data, GPU). Tuning is usually done with **Optuna**. Explanations come from **SHAP**.
- Typical uses: demand and sales forecasting, energy load, fraud detection, credit scoring, churn prediction, ad click-through, ETA prediction, and **search ranking** (LambdaMART, a gradient-boosted ranker).
- **Weather specifically:** real forecasts come from physics-based numerical weather prediction (NWP) and, increasingly, from deep-learning weather models trained on decades of global data (e.g. GraphCast, GenCast). GBDTs are popular for **post-processing**: correcting an NWP forecast for one specific station or city.
- Deployment: models are exported to **ONNX** or compiled to plain code. Prediction takes microseconds and can even run in the browser.
- For time series, **foundation models** (pre-trained on huge collections of series, e.g. TimesFM, Chronos, Moirai) and **tabular foundation models** (e.g. TabPFN) are now real competitors, especially when data is small.

### 9.3 Open problems & active debates
- **Trees can't extrapolate.** A tree predicts only values *between* those it saw in training (leaf averages). If a future winter is colder than anything in 2000–2007, or climate warming pushes temperatures above that range, the model can't predict it. Fixes include detrending, differencing, or linear-leaf trees (LightGBM `linear_tree`). This is a real risk for long-running time-series products.
- **Deep learning vs GBDT on tabular data.** Studies (e.g. Grinsztajn et al., 2022) found that trees still usually win on medium-sized tables. Tabular foundation models are challenging that, and the debate is open.
- **Uncertainty.** A good *range* matters as much as a good point, especially when errors are larger in winter. Active topics include quantile boosting, NGBoost, and **conformal prediction** (prediction ranges with mathematical guarantees).
- **Distribution shift / concept drift.** The world changes, so a model trained on 2000–2007 slowly goes stale. Questions include when to retrain and how to detect drift.
- **Trustworthy explanations.** Impurity importances are biased, and SHAP is expensive and can mislead when features are correlated.
- **Multi-step forecasting.** Recursive forecasting compounds its own errors, while direct forecasting needs many models. The best strategy is still task-dependent.

---

## 10. Why this matters for my roles

### Software engineer
- A trained model is just a **function**, `predict(features) → number`, saved as a file (`joblib`) and loaded behind an API, like any other dependency.
- **Getting data from an API is part of ML.** Here: an HTTP GET of a CSV file, keeping only the 2 columns we need out of 165, and caching the result so we don't download it every run.
- **Reproducibility = good engineering:** a separate `.venv`, pinned `requirements.txt`, fixed `random_state`, and data/feature code in one module (`common.py`) so that training and serving compute features **the same way**. A mismatch between training and serving ("training-serving skew") is a classic production bug.
- **Data leakage is the ML version of a test that reads its own expected output.** It passes, and it means nothing.

### Web / frontend developer
- Features for a live prediction must be computed from data available **at request time**. For example, yesterday's temperature must already be in your DB when the user opens the page.
- GBDTs are tiny and fast (microseconds), so they can run server-side per request or be exported to run in the browser (ONNX Runtime Web).
- The lerp/easing analogy (`x += k * (target - x)`) is a good mental model for what each boosting round does.

### UI/UX designer
- **Show uncertainty, not false precision.** "−5 °C (likely −8 to −2)" is more honest than "−5.14 °C". Make the range **wider in winter**, because the error is about 1.4× larger in February than in August.
- **Explain the prediction** in user language using feature importance or SHAP: "Cold again tonight, because last night was −10 °C."
- **Design for the misses.** 15% of days are off by more than 3 °C, and a surprise like 2009-01-09 (9.6 °C off) will be noticed. Plan for trust-repair: update often, show "last updated", and flag "unsettled weather".
- **Evaluate like an A/B test.** Always compare against a baseline ("tomorrow = today"). A model that can't beat the simple rule shouldn't ship. Here it wins, but by a small margin.

---

## 11. Glossary
| Term | Plain meaning |
|---|---|
| **Feature** | an input column the model uses (e.g. `lag_1`) |
| **Target / label** | the answer column we want to predict |
| **Regression** | predicting a number (vs. *classification*: predicting a category) |
| **Time series** | measurements in time order, where order matters |
| **Lag** | the value from N steps earlier |
| **Seasonality** | a pattern that repeats on a fixed cycle (here, yearly) |
| **Autocorrelation** | correlation of a series with a delayed copy of itself |
| **Data leakage** | the model seeing information it wouldn't have in real life (usually the future) |
| **Weak learner** | a simple model (a shallow tree) that is only slightly better than guessing |
| **Ensemble** | many models combined into one |
| **Residual** | actual − predicted |
| **Loss function** | a formula scoring how wrong the predictions are |
| **Gradient** | the slope of the loss; it points toward higher loss |
| **Learning rate (η)** | how much of each new tree we add (step size) |
| **Hyperparameter** | a setting we choose before training (depth, learning rate…) |
| **Overfitting / underfitting** | memorising noise / missing the real pattern |
| **Early stopping** | stopping when validation error stops improving |
| **Regularisation** | anything that discourages overfitting (small η, shallow trees, subsampling) |
| **OOB (out-of-bag)** | rows a tree didn't see because of subsampling; used as a mini check |
| **Baseline** | a simple non-ML rule to compare against |
| **Persistence forecast** | the baseline "tomorrow = today" |
| **MAE / RMSE / R²** | error metrics (see §7.1) |
| **Raw GitHub URL** | a direct link to a file in a GitHub repo; we request it like `fetch()` |
| **CC BY 4.0 / MIT** | open licences: free to use if you credit the source or keep the licence notice |

---

## 12. Check my understanding

<details><summary>1. Why can't we shuffle time-series data before splitting?</summary>
Shuffling puts future days into training, so the model "knows the future" and the test score becomes too optimistic (data leakage). Real forecasts only know the past.
</details>

<details><summary>2. Why is the residual the same thing as the negative gradient?</summary>
For L = ½(y−F)², dL/dF = −(y−F). The negative of that is y−F, which is the residual. So fitting trees to residuals is gradient descent on the predictions.
</details>

<details><summary>3. What happens with learning_rate = 1 and 1,000 deep trees?</summary>
Each tree fully "jumps" to fit the current errors, including noise. Training error goes near 0 but validation error rises: overfitting.
</details>

<details><summary>4. Why didn't we scale the features?</summary>
Trees split on thresholds (feature ≤ value), so they only care about the order of values, not their scale or sign.
</details>

<details><summary>5. Why use both a validation and a test set?</summary>
Validation is used to make choices, and every choice slightly "fits" to it. The test set stays untouched, so it gives an unbiased final grade.
</details>

<details><summary>6. Why does ML help less in Stockholm (≈3%) than in Melbourne (≈12%)?</summary>
In Stockholm, yesterday is a much stronger clue (autocorrelation 0.95 vs 0.77), so "tomorrow = today" is already a strong forecast. The model can only add what the baseline misses, and with past temperatures alone there isn't much left.
</details>

<details><summary>7. Could this model predict a record −35 °C night?</summary>
No. Its outputs are sums of leaf averages learned from 2000–2007 data (coldest: −19.7 °C), so it can't extrapolate beyond the range it has seen.
</details>

---

## 13. References
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics.
- Friedman, J. H. (2002). *Stochastic Gradient Boosting.* Computational Statistics & Data Analysis.
- Freund, Y. & Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting.*
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Grinsztajn, L. et al. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS.
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*, ch. 10 (Boosting).
- scikit-learn docs: `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, "Time-related feature engineering".
- **Data:** Huber, F. et al. *Weather prediction dataset* (CC BY 4.0). https://github.com/florian-huber/weather_prediction_dataset, https://doi.org/10.5281/zenodo.7525955
- **Original station data:** Klein Tank, A.M.G. and Coauthors (2002). *Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment.* Int. J. of Climatol., 22, 1441-1453. https://www.ecad.eu
