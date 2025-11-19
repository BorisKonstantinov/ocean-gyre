# train.py
import os
import json
import joblib
from typing import Optional, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.keras")
X_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "y_scaler.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")


def _artifacts_exist() -> bool:
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(X_SCALER_PATH)
        and os.path.exists(Y_SCALER_PATH)
    )


def build_model(input_dim: int, lr: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile the MLP used throughout the project.
    """
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(5),  # [t2m, d2m, sp, tcc, ssr]
        ]
    )
    model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
    return model


def train_model(
    model: tf.keras.Model,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val_scaled: np.ndarray,
    epochs: int = 100,
    batch_size: int = 20,
) -> tf.keras.callbacks.History:
    """
    Train with EarlyStopping and save-best ModelCheckpoint -> artifacts/model.keras
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=24, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cbs,
    )
    return history


def save_artifacts(
    model: tf.keras.Model,
    X_scaler,
    y_scaler,
    meta: Optional[Dict] = None,
) -> None:
    """
    Save model, scalers, and optional metadata under artifacts/.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # Save model (Keras v3 format)
    model.save(MODEL_PATH)
    # Save scalers
    joblib.dump(X_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)
    # Save metadata
    meta = {} if meta is None else dict(meta)
    meta.setdefault("saved_with", "train.py")
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_artifacts() -> Tuple[tf.keras.Model, object, object, Dict]:
    """
    Load model, scalers, and metadata from artifacts/.
    """
    if not _artifacts_exist():
        raise FileNotFoundError(
            "Artifacts not found. Train first or call get_or_train(...)."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    X_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
    meta: Dict = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    return model, X_scaler, y_scaler, meta


def get_or_train(
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val_scaled: np.ndarray,
    X_scaler,
    y_scaler,
    *,
    force_retrain: bool = False,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 20,
    meta: Optional[Dict] = None,
    input_dim: Optional[int] = None,
) -> Tuple[tf.keras.Model, object, object, Optional[tf.keras.callbacks.History]]:
    """
    If artifacts exist and force_retrain=False, load and return them.
    Otherwise build, train, save, and return the fresh model + scalers.

    Returns: (model, X_scaler, y_scaler, history)
    - history is None if the model was loaded from disk.
    """
    if _artifacts_exist() and not force_retrain:
        model, Xs, Ys, _ = load_artifacts()
        return model, Xs, Ys, None

    if input_dim is None:
        # Infer from provided arrays
        input_dim = X_train_scaled.shape[1]

    model = build_model(input_dim=input_dim, lr=lr)
    history = train_model(
        model,
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        epochs=epochs,
        batch_size=batch_size,
    )
    # Persist everything so subsequent runs can skip training
    save_artifacts(model, X_scaler, y_scaler, meta=meta)
    return model, X_scaler, y_scaler, history
