"""Sign library: one representative recorded take per word (speech -> sign figure).

Averaging a word's takes would blur the motion, so each word keeps its medoid:
the take with the smallest total distance to that word's other takes, i.e. the
most typical one. IDLE is not a sign and is left out. Saved as
``models/sign_library.npz`` ({label: (45, 153) float32}) and copied to the Pi
with the other models.
Takes for words we did not record (from MS-ASL, ``sign/extra_library.py``) are
merged in from ``models/sign_library_extra.npz`` if it exists; our own takes win.

    python -m senseless.sign.library            # data/ -> models/sign_library.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from senseless.collect import dataset
from senseless.common.config import DATA_DIR, PATHS, SIGN


def medoid_index(windows: np.ndarray) -> int:
    """Index of the take with the smallest summed Euclidean distance to the others."""
    flat = windows.reshape(len(windows), -1).astype(np.float64)
    sq = (flat**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0)
    return int(np.argmin(np.sqrt(d2).sum(axis=1)))


def build_library(
    data_dir: Path | str = DATA_DIR, exclude: tuple[str, ...] = (SIGN.idle_label,)
) -> dict[str, np.ndarray]:
    library: dict[str, np.ndarray] = {}
    for label in dataset.list_labels(data_dir):
        if label in exclude:
            continue
        windows = dataset.load_label(label, data_dir)
        if len(windows) == 0:
            continue
        library[label] = windows[medoid_index(windows)].astype(np.float32)
    return library


def save_library(library: dict[str, np.ndarray], path: Path | str = PATHS.sign_library) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **library)
    return path


def load_library(path: Path | str = PATHS.sign_library) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def add_extra(
    library: dict[str, np.ndarray], extra_path: Path | str = PATHS.sign_library_extra
) -> dict[str, np.ndarray]:
    """Add the MS-ASL takes (sign/extra_library.py) for words we did not record; ours win."""
    extra_path = Path(extra_path)
    if not extra_path.exists():
        return library
    return {**load_library(extra_path), **library}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the sign library for the Speech figure.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(PATHS.sign_library))
    args = parser.parse_args()
    library = add_extra(build_library(args.data_dir))
    path = save_library(library, args.out)
    print(f"saved {path} with {len(library)} words: {', '.join(sorted(library))}")


if __name__ == "__main__":
    main()
