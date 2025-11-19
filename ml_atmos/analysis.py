import xarray
import pandas
import numpy
import tensorflow

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

columns = ["t2m", "d2m", "sp", "tcc", "ssr"]


# --- Prediction Rollout ---
def rollout_predictions(model, X_test, y_test, steps, X_i, X_scaler, y_scaler):
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


def plot_rmse_growth(steps, xi_lim, X_test, y_test, y_train, model, X_scaler, y_scaler):
    # Average RMSE over multiple rollouts to reduce variability
    rmse = numpy.zeros((steps, 5))
    for n in range(xi_lim):
        rmse_ = rollout_predictions(
            model,
            X_test,
            y_test,
            steps=steps,
            X_i=n,
            X_scaler=X_scaler,
            y_scaler=y_scaler,
        )
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
    plt.savefig("rmse_model_vars.png")
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
    plt.ylim(0.1, 2e6)
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
    plt.ylim(-5, 5)
    plt.title(f"Skill vs Lead — {var}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.savefig(f"{figname}.png")
    return numpy.column_stack([steps, skill_pers[:, idx], skill_climo[:, idx]])
    print("works")


def training_loss(history):
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss.png")
