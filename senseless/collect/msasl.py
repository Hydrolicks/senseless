"""MS-ASL clips of words we did not record, for the Speech-mode signing figure.

    python -m senseless.collect.msasl select-anim            # manifest of the new words' clips
    python -m senseless.collect.msasl download [--limit N]   # annotated sections, yt-dlp
    python -m senseless.collect.msasl extract  [--limit N]   # needs .venv-msasl (mediapipe 0.10.18)

The working folder (default C:/Senseless_anim) holds manifest.csv, videos/ and
sequences/ (lite-tracker landmark sequences). Every step is resumable.
``sign/extra_library.py`` turns the sequences into figure takes. MS-ASL is not used
for training the sign classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
from senseless.common.config import SIGN, UI

MSASL_DIR = Path.home() / "Downloads" / "MS-ASL" / "MS-ASL"
SPLITS = ("train", "val", "test")
PAD_S = 0.5  # downloaded margin before/after the annotated sign
# A real clip of a few seconds at <=480p is >=50 KB. yt-dlp sometimes exits 0 after writing
# only an MP4 header (261 bytes, no frames) when YouTube throttles; treat that as a failure.
MIN_VIDEO_BYTES = 10_000

# Takes for the Speech-mode figure (sign/extra_library.py): MS-ASL's most frequent words
# that we did not record ourselves.
ANIM_WORK_DIR = Path("C:/Senseless_anim")
ANIM_TOP = 200  # MSASL_classes.json is ordered most frequent first
ANIM_PER_WORD = 5
_SKIP_GLOSSES = ("hoddog",)  # a typo in MS-ASL's class list


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


def _clip_from_entry(clip_id: str, label: str, split: str, entry: dict) -> Clip:
    return Clip(
        clip_id=clip_id,
        label=label,
        split=split,
        signer=int(entry["signer_id"]),
        url=entry["url"],
        start=float(entry["start_time"]),
        end=float(entry["end_time"]),
        fps=float(entry.get("fps", 30.0)),
        box=" ".join(f"{float(v):.4f}" for v in entry["box"]),
    )


def animation_glosses(classes: list[str], taken: Iterable[str], top: int = ANIM_TOP) -> list[str]:
    """MS-ASL's most frequent glosses that are new to us: single plain words only."""
    taken = set(taken)
    glosses = (c.strip().lower() for c in classes[:top])
    return [
        g
        for g in glosses
        if g not in taken and re.fullmatch(r"[a-z]+", g) and g not in _SKIP_GLOSSES
    ]


def select_animation_clips(
    entries_by_split: dict[str, list[dict]], glosses: list[str], per_word: int = ANIM_PER_WORD
) -> list[Clip]:
    """Up to ``per_word`` clips per gloss: one per signer first, then the rest, in split order."""
    rows: dict[str, list[tuple[str, int, dict]]] = {g: [] for g in glosses}
    for split in SPLITS:
        for i, entry in enumerate(entries_by_split.get(split, [])):
            gloss = str(entry["clean_text"]).strip().lower()
            if gloss in rows:
                rows[gloss].append((split, i, entry))
    clips = []
    for gloss in glosses:
        found = rows[gloss]
        firsts, signers = set(), set()
        for k, (_, _, entry) in enumerate(found):
            if int(entry["signer_id"]) not in signers:
                signers.add(int(entry["signer_id"]))
                firsts.add(k)
        order = sorted(range(len(found)), key=lambda k: (k not in firsts, k))
        for k in order[:per_word]:
            split, i, entry = found[k]
            clips.append(_clip_from_entry(f"{split}_{i:05d}", gloss.upper(), split, entry))
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
    "video is unavailable",
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
        # Plain https (DASH) formats first, H.264 preferred: section cuts of HLS formats
        # come out as empty MP4s. "b" stays as the last resort.
        "bv*[height<=480][vcodec^=avc1][protocol=https]/bv*[height<=480][protocol=https]"
        "/b[height<=480][protocol=https]/b",
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


def _has_video(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_VIDEO_BYTES


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
        if _has_video(out):
            if clip.status in ("selected", "failed"):
                clip.status, clip.note = "ok", ""
            counts["skipped"] += 1
            continue
        if limit is not None and attempts >= limit:
            break
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():  # an empty or header-only leftover
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
        if proc.returncode == 0 and _has_video(out):
            clip.status, clip.note = "ok", ""
        elif proc.returncode == 0:
            size = out.stat().st_size if out.exists() else 0
            clip.status, clip.note = "failed", f"no video in the output ({size} bytes)"
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


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MS-ASL clips of new words for the Speech-mode signing figure."
    )
    parser.add_argument("step", choices=["select-anim", "download", "extract"])
    parser.add_argument("--msasl", default=str(MSASL_DIR), help="Folder with the MS-ASL JSONs.")
    parser.add_argument(
        "--work", default=str(ANIM_WORK_DIR), help="Videos, sequences and manifest.csv."
    )
    parser.add_argument(
        "--force", action="store_true", help="Let select-anim overwrite an existing manifest."
    )
    parser.add_argument("--limit", type=int, default=None, help="At most N clips (smoke test).")
    parser.add_argument(
        "--pause", type=float, default=1.0, help="Seconds between downloads (raise if throttled)."
    )
    args = parser.parse_args()
    work = Path(args.work)
    manifest = work / "manifest.csv"

    if args.step == "select-anim":
        if manifest.exists() and not args.force:
            raise SystemExit(f"{manifest} already exists. Use --force to overwrite it.")
        entries, synonyms = load_msasl(Path(args.msasl))
        classes = json.loads((Path(args.msasl) / "MSASL_classes.json").read_text())
        glosses = animation_glosses(classes, gloss_to_label(vocabulary(), synonyms))
        clips = select_animation_clips(entries, glosses)
        write_manifest(clips, manifest)
        print(f"{len(clips)} clips of {len(glosses)} words -> {manifest}")
        return

    clips = read_manifest(manifest)
    try:
        if args.step == "download":
            counts = download_clips(
                clips, work, pause_s=args.pause, limit=args.limit, ffmpeg=_ffmpeg()
            )
        else:
            counts = extract_clips(clips, work, limit=args.limit)
        print(dict(counts))
    finally:
        write_manifest(clips, manifest)  # keep progress even after Ctrl+C


if __name__ == "__main__":
    main()
