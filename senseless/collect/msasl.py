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

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from senseless.collect import dataset
from senseless.common import landmark_schema as ls
from senseless.common.config import PROJECT_ROOT, SIGN, UI
from senseless.sign.window import default_span_s, resample_window

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
    "video is not available",
    "not available in your country",
    "not made this video available",
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
    timeout_s: float = 300.0,
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
        if out.exists() and out.stat().st_size == 0:
            out.unlink()
        try:
            proc = run(
                ytdlp_command(clip, out, ffmpeg), capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            clip.status, clip.note = "failed", f"timeout after {timeout_s:.0f} s"
            attempts += 1
            counts[clip.status] += 1
            sleep(pause_s)
            continue
        attempts += 1
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            clip.status, clip.note = "ok", ""
        else:
            err = (proc.stderr or "").strip()
            clip.status = classify_failure(err)
            clip.note = err.splitlines()[-1][:200] if err else f"exit {proc.returncode}"
        counts[clip.status] += 1
        sleep(pause_s)
    return counts


_HANDS = slice(ls.LEFT_HAND_START, ls.RIGHT_HAND_END)


def crop_box(shape: tuple, box: str, margin: float = 0.15) -> tuple[int, int, int, int]:
    """Pixel crop (top, bottom, left, right) of the signer box plus ``margin`` of its size."""
    h, w = shape[:2]
    y0, x0, y1, x1 = (float(v) for v in box.split())
    dy, dx = (y1 - y0) * margin, (x1 - x0) * margin
    top, bottom = max(0, int((y0 - dy) * h)), min(h, int(round((y1 + dy) * h)))
    left, right = max(0, int((x0 - dx) * w)), min(w, int(round((x1 + dx) * w)))
    if bottom - top < 16 or right - left < 16:
        return 0, h, 0, w
    return top, bottom, left, right


def _sane_fps(value: float) -> float:
    """A container's reported frame rate, or 30.0 when it is NaN, non-positive or absurd."""
    if math.isnan(value) or value <= 0 or value > 240:
        return 30.0
    return float(value)


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """All frames of a video as RGB arrays, and its frame rate."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = _sane_fps(float(cap.get(cv2.CAP_PROP_FPS)))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def extract_sequence(
    frames: list[np.ndarray], fps: float, box: str, backend, offset_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame feature vectors of the cropped frames; times are seconds from the sign start."""
    from senseless.sign.landmarks import frame_landmarks_to_vector

    top, bottom, left, right = crop_box(frames[0].shape, box)
    times, vecs = [], []
    for i, frame in enumerate(frames):
        crop = np.ascontiguousarray(frame[top:bottom, left:right])
        raw = backend.extract(crop, int(i * 1000 / fps))
        times.append(i / fps - offset_s)
        vecs.append(frame_landmarks_to_vector(raw))
    return np.asarray(times, dtype=np.float64), np.stack(vecs).astype(np.float32)


def sequence_path(work: Path, clip: Clip) -> Path:
    return Path(work) / "sequences" / f"{clip.clip_id}.npz"


def default_backend():
    """The Pi's lite tracker (needs mediapipe 0.10.18, i.e. .venv-msasl)."""
    from senseless.sign.landmarks import LiteBackend

    return LiteBackend()


def _extract_one(
    clip: Clip, work: Path, make_backend: Callable, reader: Callable, out: Path
) -> tuple[str, str]:
    """Extract one clip's sequence; returns its new (status, note)."""
    frames, fps = reader(video_path(work, clip))
    if not frames:
        return "no_detections", "unreadable video"
    offset = min(PAD_S, clip.start)
    with make_backend() as backend:
        times, vecs = extract_sequence(frames, fps, clip.box, backend, offset)
    duration = clip.end - clip.start
    in_sign = (times >= 0) & (times <= duration)
    if not np.any(vecs[in_sign][:, _HANDS] != 0):
        return "no_detections", "no hands during the sign"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, times=times, vecs=vecs, duration=duration)
    return "extracted", ""


def extract_clips(
    clips: list[Clip],
    work: Path,
    make_backend: Callable = default_backend,
    reader: Callable = read_frames,
    limit: int | None = None,
) -> Counter:
    """Landmark sequences for downloaded clips; a fresh tracker per clip (no carried state).

    Only ok/extracted clips are processed, so a clip that failed is not retried here.
    """
    counts: Counter = Counter()
    done = 0
    for clip in clips:
        if clip.status not in ("ok", "extracted"):
            continue
        out = sequence_path(work, clip)
        if out.exists():
            clip.status = "extracted"
            continue
        if limit is not None and done >= limit:
            break
        done += 1
        try:
            status, note = _extract_one(clip, work, make_backend, reader, out)
        except Exception as exc:  # one bad clip must not abort (or re-abort) the batch
            traceback.print_exc()
            status, note = "failed", f"extract error: {exc}"[:200]
        clip.status, clip.note = status, note
        counts[clip.status] += 1
    return counts


def onset_time(times: np.ndarray, vecs: np.ndarray, t_from: float = 0.0) -> float:
    """Time of the first frame with a hand at or after ``t_from`` (else ``t_from``)."""
    hands = np.any(vecs[:, _HANDS] != 0, axis=1)
    idx = np.flatnonzero(hands & (times >= t_from - 1e-9))
    return float(times[idx[0]]) if len(idx) else float(t_from)


def cut_windows(
    times: np.ndarray, vecs: np.ndarray, duration: float, mode: str, shifts=(0.0,)
) -> list[np.ndarray]:
    """Model windows from one sequence: onset mode (1.47 s from the first hand) or whole clip."""
    length = SIGN.window_length
    if mode == "whole":
        return [resample_window(times, vecs, end_time=duration, length=length, span_s=duration)]
    span = default_span_s()
    start = onset_time(times, vecs)
    return [
        resample_window(times, vecs, end_time=start + s + span, length=length, span_s=span)
        for s in shifts
    ]


def write_windows(clips: list[Clip], work: Path, out: Path, mode: str) -> Counter:
    """data_msasl/<split>/<LABEL>/<clip_id>_<k>.npy for every extracted clip."""
    counts: Counter = Counter()
    for clip in clips:
        seq = sequence_path(work, clip)
        if clip.status != "extracted" or not seq.exists():
            continue
        with np.load(seq) as data:
            times, vecs, duration = data["times"], data["vecs"], float(data["duration"])
        shifts = (0.0, -SHIFT_S, SHIFT_S) if clip.split == "train" else (0.0,)
        target = Path(out) / clip.split / clip.label
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob(f"{clip.clip_id}_*.npy"):  # no leftovers from another mode
            old.unlink()
        for k, window in enumerate(cut_windows(times, vecs, duration, mode, shifts)):
            np.save(target / f"{clip.clip_id}_{k}.npy", window.astype(np.float32))
            counts[clip.split] += 1
    return counts


def coverage(clips: list[Clip], out: Path) -> str:
    """Per word: annotated, downloaded, extracted (train/val/test) and train windows."""
    rows = [
        f"{'word':<10} {'annot':>5} {'downl':>5} {'train':>5} {'val':>4} {'test':>4} {'win':>5}"
    ]
    for label in sorted({c.label for c in clips}):
        mine = [c for c in clips if c.label == label]
        downloaded = sum(c.status in ("ok", "extracted", "no_detections") for c in mine)
        ext = Counter(c.split for c in mine if c.status == "extracted")
        windows = len(list((Path(out) / "train" / label).glob("*.npy")))
        flag = "  LOW" if ext["train"] < MIN_TRAIN_CLIPS else ""
        rows.append(
            f"{label:<10} {len(mine):>5} {downloaded:>5} {ext['train']:>5} {ext['val']:>4} "
            f"{ext['test']:>4} {windows:>5}{flag}"
        )
    status = Counter(c.status for c in clips)
    rows.append("status: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    return "\n".join(rows)


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    parser = argparse.ArgumentParser(description="MS-ASL clips for our vocabulary.")
    parser.add_argument("step", choices=["select", "download", "extract", "windows", "coverage"])
    parser.add_argument("--msasl", default=str(MSASL_DIR), help="Folder with the MS-ASL JSONs.")
    parser.add_argument("--work", default=str(WORK_DIR), help="Videos, sequences, manifest.")
    parser.add_argument("--out", default=str(OUT_DIR), help="Where training windows go.")
    parser.add_argument("--mode", choices=["onset", "whole"], default="onset")
    parser.add_argument(
        "--force", action="store_true", help="Let select overwrite an existing manifest."
    )
    parser.add_argument("--limit", type=int, default=None, help="At most N clips (smoke test).")
    args = parser.parse_args()
    work, manifest = Path(args.work), Path(args.work) / "manifest.csv"

    if args.step == "select":
        if manifest.exists() and not args.force:
            raise SystemExit(
                f"{manifest} already exists; select would wipe the download/extract progress. "
                "Use --force to overwrite it."
            )
        entries, synonyms = load_msasl(Path(args.msasl))
        clips = select_clips(entries, gloss_to_label(vocabulary(), synonyms))
        write_manifest(clips, manifest)
        print(f"{len(clips)} clips of {len({c.label for c in clips})} words -> {manifest}")
        return

    clips = read_manifest(manifest)
    try:
        if args.step == "download":
            print(dict(download_clips(clips, work, limit=args.limit, ffmpeg=_ffmpeg())))
        elif args.step == "extract":
            print(dict(extract_clips(clips, work, limit=args.limit)))
        elif args.step == "windows":
            print(dict(write_windows(clips, work, Path(args.out), args.mode)))
        print(coverage(clips, Path(args.out)))
    finally:
        write_manifest(clips, manifest)  # keep progress even after Ctrl+C


if __name__ == "__main__":
    main()
