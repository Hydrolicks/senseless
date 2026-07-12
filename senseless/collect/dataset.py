"""On-disk layout for collected sign samples.

Each sample is one ``(window_length, FEATURE_DIM)`` float32 window saved as a
``.npy`` under ``data/<label>/``. This module is the single definition of that
layout, so the collector (writes) and the Colab trainer (reads) agree. Pure
file I/O -- no camera, no MediaPipe -- so it is unit-tested against a temp dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.common.config import DATA_DIR, SIGN

#: Shape of one stored sample, derived from config + schema.
WINDOW_SHAPE = (SIGN.window_length, ls.FEATURE_DIM)  # (45, 153)


def label_dir(label: str, data_dir: Path | str = DATA_DIR) -> Path:
    """Directory holding the samples for ``label``."""
    return Path(data_dir) / label


def next_sample_index(label: str, data_dir: Path | str = DATA_DIR) -> int:
    """Next free integer index for a sample of ``label`` (robust to gaps)."""
    directory = label_dir(label, data_dir)
    if not directory.exists():
        return 0
    indices = []
    for path in directory.glob("*.npy"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def count_samples(label: str, data_dir: Path | str = DATA_DIR) -> int:
    """Number of stored samples for ``label`` (0 if the label has none)."""
    directory = label_dir(label, data_dir)
    return sum(1 for _ in directory.glob("*.npy")) if directory.exists() else 0


def save_window(window: np.ndarray, label: str, data_dir: Path | str = DATA_DIR) -> Path:
    """Save one window as the next sample for ``label``; return its path.

    Raises ``ValueError`` if the window is not exactly ``WINDOW_SHAPE``.
    """
    window = np.asarray(window, dtype=np.float32)
    if window.shape != WINDOW_SHAPE:
        raise ValueError(f"window shape {window.shape} != expected {WINDOW_SHAPE}")
    directory = label_dir(label, data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{next_sample_index(label, data_dir):04d}.npy"
    np.save(path, window)
    return path


def load_label(label: str, data_dir: Path | str = DATA_DIR) -> np.ndarray:
    """Load all samples for ``label`` stacked into ``(N, window_length, FEATURE_DIM)``."""
    directory = label_dir(label, data_dir)
    paths = sorted(directory.glob("*.npy")) if directory.exists() else []
    if not paths:
        return np.empty((0, *WINDOW_SHAPE), dtype=np.float32)
    return np.stack([np.load(p) for p in paths]).astype(np.float32)


def list_labels(data_dir: Path | str = DATA_DIR) -> list[str]:
    """All label names that have a sample directory under ``data_dir``."""
    base = Path(data_dir)
    if not base.exists():
        return []
    return [p.name for p in base.iterdir() if p.is_dir()]
