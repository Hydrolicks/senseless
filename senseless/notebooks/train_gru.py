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
    python -m senseless.notebooks.train_gru --lowfps-copies 0   # no low-FPS augmentation

Low-FPS augmentation: the Pi captures ~10 FPS, and the live time window
(sign/window.py) resamples those frames to the model's 45 steps. Each training
window gets ``--lowfps-copies`` extra versions simulating a camera at a random rate
between ``--min-fps`` and SIGN.reference_fps (the same resampling as inference), so
the model also learns signs as the Pi sees them. Validation and test sets are not
augmented; test accuracy is additionally reported at a simulated ``--eval-fps``.

Pi-recorded takes (``--pi-data data_pi``): takes recorded on the Pi itself (lite hand
model, ~7 FPS, the Pi's camera) close the gap the simulations can't. A share of each
word's Pi takes (``--pi-test-frac``, default 0.3) is held out and reported as the Pi
test accuracy; the rest join the training set, each counted ``--pi-weight`` times
(default 3; the Pi takes are outnumbered by the PC takes). With Pi data the low-FPS
copies are also made "Pi-like" (``--pi-like``, on by default with ``--pi-data``):
3 copies per window, 70% of them at 6-10 FPS, with short hand dropouts
(``augment.drop_hand_run``). Measured on 4 words: Pi accuracy 40% -> 91% with Pi
takes, 97.5% with Pi takes + Pi-like copies; Pi-like copies alone, without Pi takes,
lowered it, so they are off by default without ``--pi-data``.

    python -m senseless.notebooks.train_gru --pi-data data_pi
    python -m senseless.notebooks.train_gru --pi-data data_pi --pi-test-frac 0  # final model
"""

from __future__ import annotations

import argparse
import tempfile

import numpy as np

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.common.config import DATA_DIR, PATHS, SIGN
from senseless.sign.augment import drop_hand_run, perturb_hands
from senseless.sign.window import simulate_capture


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


PI_FPS_MAX = 10.0  # Pi-like copies: most are made at min_fps..PI_FPS_MAX
PI_RATE_SHARE = 0.7


def augment_low_fps(
    x: np.ndarray,
    y: np.ndarray,
    copies: int,
    min_fps: float,
    seed: int,
    hand_perturb: bool = False,
    pi_like: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Append ``copies`` simulated low-frame-rate versions of every window (training only).

    With ``hand_perturb`` each copy also gets ``augment.perturb_hands`` (small per-window
    hand rotation/scale, per-landmark bias and jitter) so the model tolerates the Pi's
    lite hand model, whose finger positions differ slightly from the recorded data.
    With ``pi_like`` each copy also gets a short hand dropout (``drop_hand_run``) and
    ``PI_RATE_SHARE`` of the copies are made at ``min_fps``-``PI_FPS_MAX`` FPS.
    """
    if copies <= 0:
        return x, y
    rng = np.random.default_rng(seed)

    def one(w: np.ndarray) -> np.ndarray:
        if hand_perturb:
            w = perturb_hands(w, rng)
        if not pi_like:
            return simulate_capture(w, fps=rng.uniform(min_fps, SIGN.reference_fps), rng=rng)
        w = drop_hand_run(w, rng)
        top = PI_FPS_MAX if rng.random() < PI_RATE_SHARE else SIGN.reference_fps
        return simulate_capture(w, fps=rng.uniform(min_fps, top), rng=rng)

    extra = [one(w) for _ in range(copies) for w in x]
    return np.concatenate([x, np.stack(extra)]).astype(np.float32), np.tile(y, copies + 1)


def load_pi_data(pi_dir: str, labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pi-recorded takes as X, y over the training ``labels``; words not in it are skipped."""
    xs, ys, skipped = [], [], []
    for label in sorted(dataset.list_labels(pi_dir)):
        if label not in labels:
            skipped.append(label)
            continue
        windows = dataset.load_label(label, pi_dir)
        xs.append(windows)
        ys.append(np.full(len(windows), labels.index(label), dtype=np.int64))
    if not xs:
        return np.empty((0, *dataset.WINDOW_SHAPE), np.float32), np.empty(0, np.int64), skipped
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys), skipped


def split_pi(
    x: np.ndarray, y: np.ndarray, test_frac: float, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Hold out ``test_frac`` of every word's Pi takes (a word with one take keeps it)."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_test = min(int(round(len(idx) * test_frac)), len(idx) - 1)
        test[idx[:n_test]] = True
    return (x[~test], y[~test]), (x[test], y[test])


def simulate_set(x: np.ndarray, fps: float, seed: int) -> np.ndarray:
    """The whole set as the live window would see it from a camera running at ``fps``."""
    rng = np.random.default_rng(seed)
    return np.stack([simulate_capture(w, fps=fps, rng=rng) for w in x]).astype(np.float32)


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
    parser.add_argument(
        "--lowfps-copies",
        type=int,
        default=None,
        help="Simulated low-FPS copies per window (default 2, or 3 with --pi-like).",
    )
    parser.add_argument("--min-fps", type=float, default=6.0, help="Lowest simulated FPS.")
    parser.add_argument("--eval-fps", type=float, default=10.0, help="Extra test-set FPS.")
    parser.add_argument(
        "--hand-perturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Perturb hand landmarks in the augmented copies (tolerate the lite hand model).",
    )
    parser.add_argument("--pi-data", default=None, help="Folder of takes recorded on the Pi.")
    parser.add_argument(
        "--pi-weight", type=int, default=3, help="Times each Pi training take is counted."
    )
    parser.add_argument(
        "--pi-test-frac",
        type=float,
        default=0.3,
        help="Share of each word's Pi takes held out as the Pi test set (0 = train on all).",
    )
    parser.add_argument(
        "--pi-like",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pi-like low-FPS copies with hand dropouts (default: on with --pi-data).",
    )
    args = parser.parse_args()
    pi_like = bool(args.pi_data) if args.pi_like is None else args.pi_like
    copies = (3 if pi_like else 2) if args.lowfps_copies is None else args.lowfps_copies

    import keras
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    keras.utils.set_random_seed(args.seed)

    x, y, labels = load_dataset(args.data_dir)
    counts = np.bincount(y, minlength=len(labels))
    pairs = ", ".join(f"{lb}={c}" for lb, c in zip(labels, counts, strict=False))
    print(f"Loaded {len(x)} samples, {len(labels)} classes.  per-label: {pairs}")
    print(f"Window shape {x.shape[1:]} (expected ({SIGN.window_length}, {ls.FEATURE_DIM}))")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed
    )
    print(f"train/val/test = {len(x_train)}/{len(x_val)}/{len(x_test)}")
    x_train, y_train = augment_low_fps(
        x_train, y_train, copies, args.min_fps, args.seed, args.hand_perturb, pi_like
    )
    if copies > 0:
        extra = " + hand perturbation" if args.hand_perturb else ""
        extra += " + Pi-like (hand dropouts, mostly <= 10 FPS)" if pi_like else ""
        print(
            f"low-FPS augmentation: {copies} copies per window at "
            f"{args.min_fps:g}-{SIGN.reference_fps:g} FPS{extra} -> {len(x_train)} training windows"
        )
    x_pi_test = np.empty((0, *dataset.WINDOW_SHAPE), np.float32)
    y_pi_test = np.empty(0, np.int64)
    if args.pi_data:
        x_pi, y_pi, skipped = load_pi_data(args.pi_data, labels)
        if skipped:
            print(f"Pi data: skipped words not in the vocabulary: {', '.join(skipped)}")
        (x_pi_train, y_pi_train), (x_pi_test, y_pi_test) = split_pi(
            x_pi, y_pi, args.pi_test_frac, args.seed
        )
        x_train = np.concatenate([x_train, np.repeat(x_pi_train, args.pi_weight, axis=0)])
        y_train = np.concatenate([y_train, np.repeat(y_pi_train, args.pi_weight)])
        print(
            f"Pi data: {len(y_pi)} takes of {len(np.unique(y_pi))} words; "
            f"{len(y_pi_train)} train (x{args.pi_weight}), {len(y_pi_test)} held out"
        )
    x_test_slow = simulate_set(x_test, args.eval_fps, args.seed)

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
    _, keras_slow = model.evaluate(x_test_slow, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\n== Held-out evaluation (Keras) ==")
    print(f"test accuracy: {keras_acc:.3f}   (simulated {args.eval_fps:g} FPS: {keras_slow:.3f})")
    label_ids = list(range(len(labels)))
    missing = sorted(set(label_ids) - set(y_test.tolist()))
    if missing:
        print(
            "NOTE: classes absent from the test split (too few samples): "
            + ", ".join(labels[i] for i in missing)
        )
    print(
        classification_report(
            y_test, y_pred, labels=label_ids, target_names=labels, zero_division=0
        )
    )
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=label_ids))
    if len(y_pi_test):
        pi_pred = np.argmax(model.predict(x_pi_test, verbose=0), axis=1)
        print("\n== Pi held-out takes (Keras) ==")
        print(f"Pi test accuracy: {np.mean(pi_pred == y_pi_test):.3f} on {len(y_pi_test)} takes")
        for i in np.unique(y_pi_test):
            hits = pi_pred[y_pi_test == i]
            wrong = ", ".join(labels[q] for q in hits[hits != i])
            note = f"  ({wrong})" if wrong else ""
            print(f"  {labels[i]:<10} {np.sum(hits == i)}/{len(hits)}{note}")

    tflite_model = export_tflite(model, x_train)
    PATHS.sign_tflite.parent.mkdir(parents=True, exist_ok=True)
    PATHS.sign_tflite.write_bytes(tflite_model)
    PATHS.sign_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")
    tfl_acc = tflite_accuracy(tflite_model, x_test, y_test)
    tfl_pi = tflite_accuracy(tflite_model, x_pi_test, y_pi_test) if len(y_pi_test) else None
    tfl_slow = tflite_accuracy(tflite_model, x_test_slow, y_test)
    print("\n== Exported TFLite ==")
    print(f"saved {PATHS.sign_tflite} ({len(tflite_model) / 1024:.0f} KB)")
    print(f"saved {PATHS.sign_labels}")
    print(
        f"tflite test accuracy: {tfl_acc:.3f}   (simulated {args.eval_fps:g} FPS: "
        f"{tfl_slow:.3f}; should track the Keras numbers)"
    )
    if tfl_pi is not None:
        print(f"tflite Pi test accuracy: {tfl_pi:.3f} on {len(y_pi_test)} held-out Pi takes")


if __name__ == "__main__":
    main()
