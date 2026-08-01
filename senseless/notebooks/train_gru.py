"""Train the sign GRU on collected data and export a TFLite model.

Smoke-test + held-out evaluation (per CLAUDE.md), not TDD. Loads every label
under ``data/`` (see ``collect.dataset``), trains a small GRU over
``(window_length, FEATURE_DIM)`` windows, reports held-out accuracy + a confusion
matrix, and exports ``models/sign_gru_int8.tflite`` + ``models/sign_labels.txt``
for the inference runner.

Runs locally on the dev PC or on Colab (import these functions in a cell, or run
as a script). TensorFlow / Keras / scikit-learn are imported lazily so importing
this module stays cheap.

    python -m senseless.notebooks.train_gru
    python -m senseless.notebooks.train_gru --epochs 80
"""

from __future__ import annotations

import argparse
import tempfile

import numpy as np

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.common.config import DATA_DIR, PATHS, SIGN


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all labels into X ``(N, T, F)`` float32, y ``(N,)`` int, and label names."""
    labels = sorted(dataset.list_labels(data_dir))
    if not labels:
        raise SystemExit(f"No data found under {data_dir}. Collect some first.")
    xs, ys = [], []
    for index, label in enumerate(labels):
        windows = dataset.load_label(label, data_dir)
        xs.append(windows)
        ys.append(np.full(len(windows), index, dtype=np.int64))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys), labels


def build_model(num_classes: int):
    """Small GRU classifier over one landmark window (< 1M params)."""
    import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(SIGN.window_length, ls.FEATURE_DIM)),
            # unroll=True: the window length is fixed, and unrolling avoids the
            # dynamic TensorList ops that TFLite can't lower for RNNs (keeps the
            # export to plain TFLite builtins, XNNPACK-friendly on the Pi).
            keras.layers.GRU(128, unroll=True),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_tflite(model, x_repr: np.ndarray) -> bytes:
    """Convert to TFLite (INT8 via a representative set; dynamic-range fallback)."""
    import tensorflow as tf

    def representative_dataset():
        for i in range(min(200, len(x_repr))):
            yield [x_repr[i : i + 1].astype(np.float32)]

    with tempfile.TemporaryDirectory() as saved_dir:
        model.export(saved_dir)  # Keras 3 -> TF SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        try:
            return converter.convert()
        except Exception as exc:  # GRU full-INT8 can be finicky; keep the slice working
            print(f"INT8 conversion failed ({type(exc).__name__}); using dynamic-range.")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            return converter.convert()


def tflite_accuracy(tflite_model: bytes, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of the exported TFLite model on the test set (validates the artifact)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    correct = 0
    for x, y in zip(x_test, y_test, strict=False):
        interpreter.set_tensor(inp["index"], x[None].astype(inp["dtype"]))
        interpreter.invoke()
        if int(np.argmax(interpreter.get_tensor(out["index"])[0])) == int(y):
            correct += 1
    return correct / len(x_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + export the sign GRU.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import keras
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    keras.utils.set_random_seed(args.seed)

    x, y, labels = load_dataset(args.data_dir)
    print(f"Loaded {len(x)} samples, {len(labels)} classes: {labels}")
    print(f"Window shape {x.shape[1:]} (expected ({SIGN.window_length}, {ls.FEATURE_DIM}))")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed
    )
    print(f"train/val/test = {len(x_train)}/{len(x_val)}/{len(x_test)}")

    model = build_model(len(labels))
    model.summary()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[stop],
        verbose=2,
    )

    _, keras_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\n== Held-out evaluation (Keras) ==")
    print(f"test accuracy: {keras_acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    tflite_model = export_tflite(model, x_train)
    PATHS.sign_tflite.parent.mkdir(parents=True, exist_ok=True)
    PATHS.sign_tflite.write_bytes(tflite_model)
    PATHS.sign_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")
    tfl_acc = tflite_accuracy(tflite_model, x_test, y_test)
    print("\n== Exported TFLite ==")
    print(f"saved {PATHS.sign_tflite} ({len(tflite_model) / 1024:.0f} KB)")
    print(f"saved {PATHS.sign_labels}")
    print(f"tflite test accuracy: {tfl_acc:.3f} (should track the Keras number)")


if __name__ == "__main__":
    main()
