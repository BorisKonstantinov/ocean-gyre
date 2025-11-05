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
    ) #Its actually 23' and 24'
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
X = df[["t2m", "d2m", "sp", "tcc", "ssr", "hour_sin", "hour_cos", "day_sin", "day_cos"]]
# X = df[["t2m", "d2m", "sp", "tcc", "ssr"]]
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
model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error")
model.summary()
# Train the model as in pseudocode loop
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=32,
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
    predictions = []
    rmse = []
    current_state = X_test.iloc[X_i].copy()
    current_time = X_test.index[X_i]

    for step in range(steps):
        x_scaled = X_scaler.transform(
            pandas.DataFrame([current_state], columns=X_test.columns)
        )
        y_scaled_pred = model.predict(x_scaled, verbose=0)
        pred = y_scaler.inverse_transform(y_scaled_pred)[0]  # Back to physical units
        predictions.append(pred)

        actual_idx = X_i + (step) * 6
        if actual_idx < len(y_test):
            actual = y_test[actual_idx]
            err = numpy.sqrt((actual - pred) ** 2)
            rmse.append(err)
            # names = ["t2m", "d2m", "sp", "tcc", "ssr"]
            # format each RMSE as a Python float with 3 decimal places
            # formatted = ", ".join(f"{n}: {float(e):.3f}" for n, e in zip(names, err))
            # print(f"Step {step+1}: RMSEs -> {formatted}")

        current_time += pandas.Timedelta(hours=6)
        new_state = current_state.copy()
        new_state[["t2m", "d2m", "sp", "tcc", "ssr"]] = pred
        new_state["hour_sin"] = numpy.sin(2 * numpy.pi * current_time.hour / 24)
        new_state["hour_cos"] = numpy.cos(2 * numpy.pi * current_time.hour / 24)
        new_state["day_sin"] = numpy.sin(2 * numpy.pi * current_time.dayofyear / 365.25)
        new_state["day_cos"] = numpy.cos(2 * numpy.pi * current_time.dayofyear / 365.25)
        current_state = new_state
    return numpy.array(predictions), numpy.array(rmse)

def plot_rmse_growth(stps, xi_lim, figname):
    # Average RMSE over multiple rollouts to reduce variability
    rmse = numpy.zeros((stps, 5))
    for n in range(xi_lim):
        preds, rmse_ = rollout_predictions(model, X_test, y_test, steps=stps, X_i=n)
        rmse += rmse_
    rmse /= xi_lim
    # Plot RMSE growth over steps
    plt.figure(figsize=(10, 6))
    steps = numpy.arange(1, rmse.shape[0] + 1)
    for i, name in enumerate(columns):
        plt.plot(steps, rmse[:, i], marker='o', label=name)
    plt.yscale('log')
    plt.xlabel('Steps (6h each)')
    plt.ylabel('RMSE')
    plt.title('RMSE Growth over Autoregressive Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{figname}.png')
    return rmse

rmse = plot_rmse_growth(40, 50, "rmse_6")