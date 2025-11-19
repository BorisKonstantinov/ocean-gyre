# Mahchine Learning training to predict T(t=t0+6h)=f(t=t0)
"""Side project built in preparation for Met Office interview.

Option 3: You're training a weather prediction model that takes atmospheric state at
time t and predicts state at t+6h. Your training loop is (in pseudocode):

for each sample (state_t, state_t6h) in dataset:
  prediction = model(state_t)
  loss = MSE(prediction, state_t6h)
  backpropagate(loss)

Training converges well. Single-step validation error (predicting 6h ahead) is 0.8 K for
temperature.However, when you roll the model out autoregressively for 10 steps (60
hours):
• After 12h (2 steps):  error ~1.7 K
• After 24h (4 steps):  error ~4.2 K
• After 60h (10 steps): error ~18.5 K

Part A: Explain why the multi-step error grows much faster than linear accumulation of
the single-step error would suggest.
Part B: Propose a modification to the training procedure to address this. What
trade-offs does your approach involve?
"""

import xarray
import pandas
import numpy

from sklearn.preprocessing import StandardScaler

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

import matplotlib.pyplot as plt


# --- Load Data ---
def load_data():
    # Load data individually per variable
    ds_t2m = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "2t"}
    )  # Its actually 23' and 24'
    ds_d2m = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "2d"}
    )
    ds_sp = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "sp"}
    )
    ds_tcc = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "tcc"}
    )
    ds_ssr = xarray.open_dataset(
        "2425Ex.grib", engine="cfgrib", filter_by_keys={"shortName": "ssr"}
    )
    # Convert to Pandas DataFrame
    df_t2m = ds_t2m.to_dataframe()
    df_d2m = ds_d2m.to_dataframe()
    df_sp = ds_sp.to_dataframe()
    df_tcc = ds_tcc.to_dataframe()
    df_ssr = ds_ssr.to_dataframe()
    # Drop clutter
    df_t2m = df_t2m.drop(columns=["number", "step", "surface", "valid_time"])
    df_d2m = df_d2m.drop(columns=["number", "step", "surface", "valid_time"])
    df_sp = df_sp.drop(columns=["number", "step", "surface", "valid_time"])
    df_tcc = df_tcc.drop(columns=["number", "step", "surface", "valid_time"])
    # Drop lat/lon from index
    df_t2m.index = df_t2m.index.droplevel(["latitude", "longitude"])
    df_d2m.index = df_d2m.index.droplevel(["latitude", "longitude"])
    df_sp.index = df_sp.index.droplevel(["latitude", "longitude"])
    df_tcc.index = df_tcc.index.droplevel(["latitude", "longitude"])
    # Solar Radiation needs special handling
    df_ssr = df_ssr[["valid_time", "ssr"]].reset_index(drop=True)
    df_ssr = df_ssr.set_index("valid_time")
    df_ssr.index.name = "time"
    df_ssr = df_ssr[(df_ssr.index >= "2023-01-01") & (df_ssr.index < "2024-12-30")]
    # Merge all dataframes into a master
    df = pandas.concat([df_t2m, df_d2m, df_sp, df_tcc, df_ssr], axis=1)
    df = df[(df.index >= "2023-01-01") & (df.index < "2024-12-29")]
    # Preview data information
    print("Description \n", df.describe(), "\n", "-" * 70)
    print("Correlation matrix \n", df.corr(numeric_only=True), "\n", "-" * 70)
    print("Data head \n", df.head(), "\n", "-" * 70)
    return df


df = load_data()


# --- Structure Data ---
# Convert T[K] -> T[C]
df["t2m"] = df["t2m"] - 273.15
df["d2m"] = df["d2m"] - 273.15
# Make time cyclical
df["hour_sin"] = numpy.sin(2 * numpy.pi * df.index.hour / 24.0)
df["hour_cos"] = numpy.cos(2 * numpy.pi * df.index.hour / 24.0)
df["day_sin"] = numpy.sin(2 * numpy.pi * df.index.dayofyear / 365.25)
df["day_cos"] = numpy.cos(2 * numpy.pi * df.index.dayofyear / 365.25)
# Define features X and prediction y(X)
X = df[
    ["t2m", "d2m", "sp", "tcc", "ssr"]
]  # , "hour_sin", "hour_cos", "day_sin", "day_cos"]]
y = df[["t2m", "d2m", "sp", "tcc", "ssr"]].shift(-6)
# Reshape X and y to match
X = X.iloc[:-6]
y = y.iloc[:-6]
# Create a cronological testing & training split
split = int(0.79 * len(X))
X_train = X.iloc[:split]
y_train = y.iloc[:split]
X_test = X.iloc[split:]
y_test = y.iloc[split:]
# Scale the data with mean 0, std 1
X_scaler = StandardScaler()
y_scaler = StandardScaler()
# Test data is scaled with training data fit
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)
# Convert y to a NumPy array from a DataFrame
y_train = y_train.values
y_test = y_test.values


# --- Build Model ---
# A simple feed-forward neural network (MLP)
model = Sequential(
    [
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(5),
    ]
)
# loss = MSE(prediction, state_t6h)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
model.summary()
# Train the model as in pseudocode loop
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=20,
    validation_data=(X_test_scaled, y_test_scaled),
    verbose=1,
)
# Evaluate
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
# Compute per-variable RMSE
rmse = numpy.sqrt(((y_pred - y_test) ** 2).mean(axis=0))
columns = ["t2m", "d2m", "sp", "tcc", "ssr"]
for name, val in zip(columns, rmse):
    print(f"Test RMSE for {name}: {val:.3f}")


# --- Prediction Rollout ---
def rollout_predictions(
    model, X_test, y_test, steps=10, X_i=0, X_scaler=X_scaler, y_scaler=y_scaler
):
    rmse = []
    current_state = X_test.iloc[X_i].copy()
    # current_time = X_test.index[X_i]

    for step in range(steps):
        x_scaled = X_scaler.transform(
            pandas.DataFrame([current_state], columns=X_test.columns)
        )
        y_scaled_pred = model.predict(x_scaled, verbose=0)
        y_scaled_pred += numpy.random.normal(0, 0.05, size=y_scaled_pred.shape)
        pred = y_scaler.inverse_transform(y_scaled_pred)[0]

        actual_idx = X_i + (step) * 6
        if actual_idx < len(y_test):
            actual = y_test[actual_idx]
            err = (actual - pred) ** 2
            rmse.append(err)

        # current_time += pandas.Timedelta(hours=6)
        new_state = current_state.copy()
        new_state[["t2m", "d2m", "sp", "tcc", "ssr"]] = pred
        # new_state["hour_sin"] = numpy.sin(2 * numpy.pi * current_time.hour / 24)
        # new_state["hour_cos"] = numpy.cos(2 * numpy.pi * current_time.hour / 24)
        # new_state["day_sin"] = numpy.sin(2 * numpy.pi * current_time.dayofyear / 365.25)
        # new_state["day_cos"] = numpy.cos(2 * numpy.pi * current_time.dayofyear / 365.25)
        current_state = new_state
    return numpy.array(rmse)


def plot_rmse_growth(stps, xi_lim, figname):
    # Average RMSE over multiple rollouts to reduce variability
    rmse = numpy.zeros((stps, 5))
    for n in range(xi_lim):
        rmse_ = rollout_predictions(model, X_test, y_test, steps=stps, X_i=n)
        rmse += rmse_
    rmse = numpy.sqrt(rmse / xi_lim)
    # Plot RMSE growth over steps
    plt.figure(figsize=(10, 6))
    steps = numpy.arange(1, rmse.shape[0] + 1)
    for i, name in enumerate(columns):
        plt.plot(steps, rmse[:, i], marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("Steps (6h each)")
    plt.ylabel("RMSE")
    plt.title("RMSE Growth over Autoregressive Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{figname}.png")
    return rmse


# Baselines & Skill Curves
def compute_skill_curves(
    steps, xi_lim, X_test, y_test, y_train, model, X_scaler, y_scaler
):
    """
    Returns:
      rmse_model        : (steps, 5)
      rmse_persistence  : (steps, 5)
      rmse_climatology  : (steps, 5)
      skill_vs_persist  : (steps, 5)  -> 1 - RMSE_model / RMSE_persistence
      skill_vs_climo    : (steps, 5)  -> 1 - RMSE_model / RMSE_climatology
    """

    # Ensure we have enough room to compare at the longest lead
    max_xi = min(xi_lim, (len(y_test) - 1) // (6 * steps))
    if max_xi < 1:
        raise ValueError("Not enough test samples for the requested steps.")

    # Climatology mean (vector of 5 in physical units)
    climo = y_train.mean(axis=0)

    # Accumulators for MSE (we'll sqrt at the end)
    mse_model = numpy.zeros((steps, 5), dtype=float)
    mse_pers = numpy.zeros((steps, 5), dtype=float)
    mse_climo = numpy.zeros((steps, 5), dtype=float)

    for n in range(max_xi):
        # Model rollout from start index n (aggregates abs error per step)
        se_model = rollout_predictions(
            model,
            X_test,
            y_test,
            steps=steps,
            X_i=n,
            X_scaler=X_scaler,
            y_scaler=y_scaler,
        )

        # Persistence baseline: predict the t0 state forever
        x0 = X_test.iloc[n][["t2m", "d2m", "sp", "tcc", "ssr"]].values
        for s in range(steps):
            actual_idx = n + s * 6
            actual = y_test[actual_idx]
            se_p = (x0 - actual) ** 2
            se_c = (climo - actual) ** 2
            mse_pers[s] += se_p
            mse_climo[s] += se_c

        mse_model += se_model

    # Mean over starts, then sqrt -> RMSE
    rmse_model = numpy.sqrt(mse_model / max_xi)
    rmse_persistence = numpy.sqrt(mse_pers / max_xi)
    rmse_climatology = numpy.sqrt(mse_climo / max_xi)

    # Skill: 1 - RMSE_model / RMSE_baseline (safe divide)
    eps = 1e-12
    skill_vs_persist = 1.0 - rmse_model / (rmse_persistence + eps)
    skill_vs_climo = 1.0 - rmse_model / (rmse_climatology + eps)

    return (
        rmse_model,
        rmse_persistence,
        rmse_climatology,
        skill_vs_persist,
        skill_vs_climo,
    )


def plot_error_curves(rmse_model, rmse_pers, rmse_climo, var="t2m", figname="error"):
    """
    Plot RMSE vs lead (steps) for a single variable on log scale.
    Returns an array [step, rmse_model, rmse_persistence, rmse_climatology].
    """
    idx = columns.index(var)
    steps = numpy.arange(1, rmse_model.shape[0] + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, rmse_model[:, idx], marker="o", label="Model")
    plt.plot(steps, rmse_pers[:, idx], marker="s", label="Persistence")
    plt.plot(steps, rmse_climo[:, idx], marker="^", label="Climatology")
    plt.yscale("log")
    plt.xlabel("Steps (6h each)")
    plt.ylabel("RMSE (native units)")
    plt.ylim(-1e-12, 2e6)
    plt.title(f"RMSE vs Lead — {var}")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.savefig(f"{figname}.png")
    return numpy.column_stack(
        [steps, rmse_model[:, idx], rmse_pers[:, idx], rmse_climo[:, idx]]
    )


def plot_skill_curves(skill_pers, skill_climo, var="t2m", figname="skill"):
    """
    Plot skill vs lead for a single variable.
    Skill = 1 - RMSE(model)/RMSE(baseline). Higher is better; 0 = same as baseline.
    Returns an array [step, skill_vs_persistence, skill_vs_climatology].
    """
    idx = columns.index(var)
    steps = numpy.arange(1, skill_pers.shape[0] + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, skill_pers[:, idx], marker="o", label="Skill vs Persistence")
    plt.plot(steps, skill_climo[:, idx], marker="s", label="Skill vs Climatology")
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Steps (6h each)")
    plt.ylabel("Skill (1 - RMSE_model/RMSE_baseline)")
    plt.ylim(-5.5, 4.5)
    plt.title(f"Skill vs Lead — {var}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.savefig(f"{figname}.png")
    return numpy.column_stack([steps, skill_pers[:, idx], skill_climo[:, idx]])


rmse_model, rmse_pers, rmse_climo, skill_pers, skill_climo = compute_skill_curves(
    steps=30,
    xi_lim=100,
    X_test=X_test,
    y_test=y_test,
    y_train=y_train,
    model=model,
    X_scaler=X_scaler,
    y_scaler=y_scaler,
)

tbl_err_t2m = plot_error_curves(rmse_model, rmse_pers, rmse_climo, var="t2m")
tbl_skill_t2m = plot_skill_curves(skill_pers, skill_climo, var="t2m")
print(tbl_err_t2m[:5])  # first 5 rows
print(tbl_skill_t2m[:5])


rmse = plot_rmse_growth(30, 100, "rmse_multi")
