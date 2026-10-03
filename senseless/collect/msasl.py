"""MS-ASL clips for our vocabulary: select, download, extract landmarks, cut windows.

    python -m senseless.collect.msasl select                 # manifest of our words' clips
    python -m senseless.collect.msasl download [--limit N]   # annotated sections, yt-dlp
    python -m senseless.collect.msasl extract  [--limit N]   # needs .venv-msasl (mediapipe 0.10.18)
    python -m senseless.collect.msasl windows  [--mode onset|whole] [--out DIR]
    python -m senseless.collect.msasl coverage

The working folder (default C:/Senseless_msasl) holds manifest.csv, videos/ and
sequences/; training windows go to data_msasl/<split>/<LABEL>/ in the repo root, in
the same (45, 153) format as data/. Every step is resumable.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from senseless.collect import dataset
from senseless.common.config import PROJECT_ROOT, SIGN, UI

MSASL_DIR = Path.home() / "Downloads" / "MS-ASL" / "MS-ASL"
WORK_DIR = Path("C:/Senseless_msasl")
OUT_DIR = PROJECT_ROOT / "data_msasl"
SPLITS = ("train", "val", "test")
PAD_S = 0.5  # downloaded margin before/after the annotated sign
SHIFT_S = 0.15  # train-split windows also start this much earlier/later
MIN_TRAIN_CLIPS = 15  # fewer extracted train clips than this flags the word


@dataclass
class Clip:
    clip_id: str
    label: str
    split: str
    signer: int
    url: str
    start: float
    end: float
    fps: float
    box: str  # "y0 x0 y1 x1", normalized to the frame
    status: str = "selected"  # selected | ok | unavailable | failed | extracted | no_detections
    note: str = ""


_TYPES = {f.name: f.type for f in fields(Clip)}


def vocabulary(data_dir: Path | str = dataset.DATA_DIR) -> list[str]:
    """Our word labels (everything recorded under data/ except the IDLE class)."""
    return sorted(lb for lb in dataset.list_labels(data_dir) if lb != SIGN.idle_label)


def gloss_to_label(
    vocab: Iterable[str],
    synonyms: Iterable[Iterable[str]],
    aliases: tuple[tuple[str, str], ...] = UI.sign_aliases,
) -> dict[str, str]:
    """MS-ASL gloss text (lowercase) -> our label."""
    vocab = set(vocab)
    mapping = {label.lower(): label for label in vocab}
    for phrase, label in aliases:
        if label in vocab:
            mapping[phrase] = label
    for group in synonyms:
        group = [g.strip().lower() for g in group]
        hits = {mapping[g] for g in group if g in mapping}
        if len(hits) == 1:
            label = hits.pop()
            for g in group:
                mapping.setdefault(g, label)
    return mapping


def select_clips(entries_by_split: dict[str, list[dict]], mapping: dict[str, str]) -> list[Clip]:
    """One Clip per MS-ASL entry whose gloss maps to our vocabulary."""
    clips = []
    for split in SPLITS:
        for i, entry in enumerate(entries_by_split.get(split, [])):
            label = mapping.get(str(entry["clean_text"]).strip().lower())
            if label is None:
                continue
            clips.append(
                Clip(
                    clip_id=f"{split}_{i:05d}",
                    label=label,
                    split=split,
                    signer=int(entry["signer_id"]),
                    url=entry["url"],
                    start=float(entry["start_time"]),
                    end=float(entry["end_time"]),
                    fps=float(entry.get("fps", 30.0)),
                    box=" ".join(f"{float(v):.4f}" for v in entry["box"]),
                )
            )
    return clips


def write_manifest(clips: list[Clip], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f.name for f in fields(Clip)])
        for clip in clips:
            writer.writerow(astuple(clip))


def read_manifest(path: Path) -> list[Clip]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    casts = {"int": int, "float": float, "str": str}
    return [Clip(**{k: casts[_TYPES[k]](v) for k, v in row.items()}) for row in rows]


def load_msasl(msasl_dir: Path) -> tuple[dict[str, list[dict]], list[list[str]]]:
    entries = {s: json.loads((msasl_dir / f"MSASL_{s}.json").read_text()) for s in SPLITS}
    synonyms = json.loads((msasl_dir / "MSASL_synonym.json").read_text())
    return entries, synonyms


_UNAVAILABLE = (
    "private video",
    "video unavailable",
    "has been removed",
    "account associated",
    "not available",
    "copyright",
    "members-only",
    "confirm your age",
    "terminated",
)


def video_path(work: Path, clip: Clip) -> Path:
    return Path(work) / "videos" / f"{clip.clip_id}.mp4"


def ytdlp_command(clip: Clip, out: Path, ffmpeg: str | None = None) -> list[str]:
    """yt-dlp call fetching only [start - PAD_S, end + PAD_S] at <= 480p, video only."""
    a, b = max(0.0, clip.start - PAD_S), clip.end + PAD_S
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f",
        "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[height<=480]/b",
        "--download-sections",
        f"*{a:.2f}-{b:.2f}",
        "--force-keyframes-at-cuts",
        "--remux-video",
        "mp4",
    ]
    if ffmpeg:
        cmd += ["--ffmpeg-location", ffmpeg]
    return [*cmd, "-o", str(out), clip.url]


def classify_failure(stderr: str) -> str:
    text = stderr.lower()
    return "unavailable" if any(k in text for k in _UNAVAILABLE) else "failed"


def download_clips(
    clips: list[Clip],
    work: Path,
    run: Callable = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 1.0,
    limit: int | None = None,
    ffmpeg: str | None = None,
) -> Counter:
    """Fetch every clip not yet on disk (unavailable ones are not retried)."""
    counts: Counter = Counter()
    attempts = 0
    for clip in clips:
        if clip.status == "unavailable":
            counts["unavailable"] += 1
            continue
        out = video_path(work, clip)
        if out.exists() and out.stat().st_size > 0:
            if clip.status in ("selected", "failed"):
                clip.status, clip.note = "ok", ""
            counts["skipped"] += 1
            continue
        if limit is not None and attempts >= limit:
            break
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = run(ytdlp_command(clip, out, ffmpeg), capture_output=True, text=True)
        attempts += 1
        if proc.returncode == 0 and out.exists():
            clip.status, clip.note = "ok", ""
        else:
            err = (proc.stderr or "").strip()
            clip.status = classify_failure(err)
            clip.note = err.splitlines()[-1][:200] if err else f"exit {proc.returncode}"
        counts[clip.status] += 1
        sleep(pause_s)
    return counts
