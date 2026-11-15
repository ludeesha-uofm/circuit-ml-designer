"""Neural network that designs a ladder circuit from a specification.

The network takes two numbers (the wanted gain and input resistance, both as
log10 values) and outputs the log10 of every resistor in the ladder.

Two ways of training the same architecture are compared:

1. SUPERVISED: learn to copy the resistor values stored in the dataset.
   Many different circuits give the same gain and input resistance, so this
   task is ambiguous and the network tends to predict an "average" circuit.

2. PHYSICS-INFORMED: the loss is how far the predicted circuit's gain and input
   resistance are from the target, computed by a differentiable version of the
   ladder equations. Any circuit that meets the specification is rewarded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .config import LOAD_OHMS, R_MAX, R_MIN
from .data_generator import column_names

LOG_MIN, LOG_MAX = float(np.log10(R_MIN)), float(np.log10(R_MAX))
LN10 = float(np.log(10.0))


# ----------------------------------------------------------------- data helpers
def spec_features(df: pd.DataFrame) -> np.ndarray:
    """Model input: [log10(gain), log10(input resistance)]."""
    return np.column_stack([np.log10(df["gain"]), np.log10(df["rin_ohms"])]).astype("float32")


def resistor_targets(df: pd.DataFrame, stages: int) -> np.ndarray:
    """Supervised targets: log10 of each resistor (series first, then shunt)."""
    series_cols, shunt_cols = column_names(stages)
    return np.log10(df[series_cols + shunt_cols].to_numpy()).astype("float32")


# ------------------------------------------------------ differentiable physics
class LadderPhysics(keras.layers.Layer):
    """Closed-form ladder response in TensorFlow, so gradients can flow through it.

    Input:  log10 resistors, shape (batch, 2 * stages)
    Output: [log10(gain), log10(input resistance)], shape (batch, 2)
    """

    def __init__(self, stages: int, load: float = LOAD_OHMS, **kwargs):
        super().__init__(**kwargs)
        self.stages = stages
        self.load = load

    def call(self, log_r):
        r = tf.pow(10.0, log_r)
        series, shunt = r[:, : self.stages], r[:, self.stages:]

        def parallel(x, y):
            return x * y / (x + y)

        z = parallel(shunt[:, -1], self.load)
        zs = [z]
        for k in range(self.stages - 2, -1, -1):
            z = parallel(shunt[:, k], series[:, k + 1] + z)
            zs.append(z)
        zs.reverse()

        gain = tf.ones_like(zs[0])
        for k in range(self.stages):
            gain = gain * zs[k] / (series[:, k] + zs[k])
        rin = series[:, 0] + zs[0]
        return tf.stack([tf.math.log(gain) / LN10, tf.math.log(rin) / LN10], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"stages": self.stages, "load": self.load})
        return config


# ------------------------------------------------------------------- the model
def build_design_model(stages: int, spec_sample: np.ndarray, hidden=(128, 128, 64)) -> keras.Model:
    """Specification -> log10 resistor values, limited to the allowed resistor range."""
    normalizer = keras.layers.Normalization(axis=-1, name="normalize_spec")
    normalizer.adapt(spec_sample)

    inputs = keras.Input(shape=(2,), name="spec")
    x = normalizer(inputs)
    for units in hidden:
        x = keras.layers.Dense(units, activation="swish")(x)
    x = keras.layers.Dense(2 * stages, activation="sigmoid")(x)
    outputs = keras.layers.Rescaling(scale=LOG_MAX - LOG_MIN, offset=LOG_MIN,
                                     name="log10_resistors")(x)
    return keras.Model(inputs, outputs, name="circuit_designer")


def _callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]


def train_supervised(model: keras.Model, train_df, val_df, stages: int,
                     epochs: int = 200, batch_size: int = 256, lr: float = 1e-3, verbose: int = 0):
    """Train to copy the dataset's resistor values (mean squared error in log10 space)."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model.fit(
        spec_features(train_df), resistor_targets(train_df, stages),
        validation_data=(spec_features(val_df), resistor_targets(val_df, stages)),
        epochs=epochs, batch_size=batch_size, callbacks=_callbacks(), verbose=verbose,
    )


def train_physics(model: keras.Model, train_df, val_df, stages: int, load: float = LOAD_OHMS,
                  epochs: int = 200, batch_size: int = 256, lr: float = 1e-3, verbose: int = 0):
    """Train so that the predicted circuit meets the specification (resistor labels are not used)."""
    physics_model = keras.Model(model.input, LadderPhysics(stages, load)(model.output))
    physics_model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    x_train, x_val = spec_features(train_df), spec_features(val_df)
    return physics_model.fit(
        x_train, x_train, validation_data=(x_val, x_val),
        epochs=epochs, batch_size=batch_size, callbacks=_callbacks(), verbose=verbose,
    )
