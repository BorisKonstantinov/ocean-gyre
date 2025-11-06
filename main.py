import xarray
import pandas
import numpy
import tensorflow

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

# Homebrew imports
import data
import analysis


df = data.load_data()
X, y = data.structure_data(df)
(
    X_train,
    y_train,
    X_test,
    y_test,
    X_scaler,
    y_scaler,
    X_train_scaled,
    X_test_scaled,
    y_train_scaled,
    y_test_scaled,
) = data.prep_data(X, y)


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
# Plot training loss
analysis.training_loss(history)
# Evaluate
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
# Compute per-variable RMSE
rmse = numpy.sqrt(((y_pred - y_test) ** 2).mean(axis=0))
columns = ["t2m", "d2m", "sp", "tcc", "ssr"]
for name, val in zip(columns, rmse):
    print(f"Test RMSE for {name}: {val:.3f}")

rmse_model, rmse_pers, rmse_climo, skill_pers, skill_climo = (
    analysis.compute_skill_curves(
        steps=30,
        xi_lim=100,
        X_test=X_test,
        y_test=y_test,
        y_train=y_train,
        model=model,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )
)

tbl_err_t2m = analysis.plot_error_curves(
    rmse_model=rmse_model,
    rmse_pers=rmse_pers,
    rmse_climo=rmse_climo,
    var="t2m",
    figname="error",
)
tbl_skill_t2m = analysis.plot_skill_curves(
    skill_pers=skill_pers, skill_climo=skill_climo, var="t2m", figname="skill"
)
