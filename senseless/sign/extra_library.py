"""MS-ASL takes for the Speech-mode figure: clean, rank, review, add to the library.

Words we did not record get a stick-figure take from MS-ASL clips (lite-tracker
landmark sequences from ``collect.msasl``). Each clip is trimmed to the annotated
sign and then to its active part (the idle lead-in and tail go), dropped if the
hands are missing in most frames, has short tracker gaps interpolated, is mirrored
to right-hand dominance like our own takes, and is resampled to 30 FPS at its
natural speed; a sign longer than 3 s is sped up to fit 3 s, never cut. Per word,
the most typical clip is the automatic pick; a review page lets a person reject
words or pick alternates (choices name clip ids, so they survive a re-run of
``review``). Our own recorded takes always win (``sign/library.py``).

Data source: the takes are derived from MS-ASL (Vaezi Joze and Koller, Microsoft,
BMVC 2019), used under the Computational Use of Data Agreement (C-UDA). Only
landmark skeletons are stored or shown: no video and no faces. The agreement ships
with the dataset download as ``C-UDA-0.1_annotated_discussion.pdf``.

    python -m senseless.sign.extra_library review   # C:/Senseless_anim -> review.html
    python -m senseless.sign.extra_library build [--choices choices.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from senseless.common import landmark_schema as ls
from senseless.sign.window import resample_window

FPS = 30.0  # playback rate of the figure (UI.figure_tick_ms ~ 33 ms)
MAX_TAKE_S = 3.0
ACTIVE_MOTION_REL = 0.15  # active frame: hand motion >= this share of the 90th percentile
ACTIVE_PAD_S = 0.15  # context kept before the first and after the last active frame
MIN_HAND_COVERAGE = 0.6
MAX_GAP_S = 0.3
COMPARE_STEPS = 45  # takes are resampled to this many steps only to compare them

_LEFT = slice(ls.LEFT_HAND_START, ls.LEFT_HAND_END)
_RIGHT = slice(ls.RIGHT_HAND_START, ls.RIGHT_HAND_END)
_POSE = slice(ls.POSE_START, ls.POSE_END)
_BLOCKS = (_LEFT, _RIGHT, _POSE)
_C = ls.COORDS_PER_LANDMARK
_POSE_PAIRS = tuple(
    (i, ls.POSE_LANDMARK_NAMES.index("right_" + name[len("left_") :]))
    for i, name in enumerate(ls.POSE_LANDMARK_NAMES)
    if name.startswith("left_")
)


def _present(vecs: np.ndarray, block: slice) -> np.ndarray:
    return np.any(vecs[:, block] != 0.0, axis=1)


def hand_coverage(vecs: np.ndarray) -> float:
    """Share of frames with at least one hand."""
    if len(vecs) == 0:
        return 0.0
    return float(np.mean(_present(vecs, _LEFT) | _present(vecs, _RIGHT)))


def fill_gaps(times: np.ndarray, vecs: np.ndarray, max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """Linearly fill a block missing between two frames that have it, if they are <= max_gap_s apart."""  # noqa: E501
    out = np.array(vecs, dtype=np.float32, copy=True)
    for block in _BLOCKS:
        idx = np.flatnonzero(_present(out, block))
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            if b - a > 1 and times[b] - times[a] <= max_gap_s + 1e-9:
                w = ((times[a + 1 : b] - times[a]) / (times[b] - times[a]))[:, None]
                out[a + 1 : b, block] = out[a, block] * (1.0 - w) + out[b, block] * w
    return out


def motion(vecs: np.ndarray, block: slice) -> float:
    """Summed frame-to-frame change of a block, over consecutive frames that both have it."""
    present = _present(vecs, block)
    both = present[1:] & present[:-1]
    return float(np.abs(np.diff(vecs[:, block], axis=0))[both].sum())


def mirror(vecs: np.ndarray) -> np.ndarray:
    """Left-right mirror: negate x, swap the hands and the left/right pose points."""
    out = np.array(vecs, dtype=np.float32, copy=True)
    out[:, 0::_C] *= -1.0  # every point starts on a multiple of 3: x of every point
    out[:, _LEFT], out[:, _RIGHT] = out[:, _RIGHT].copy(), out[:, _LEFT].copy()
    for a, b in _POSE_PAIRS:
        pa = slice(ls.POSE_START + a * _C, ls.POSE_START + (a + 1) * _C)
        pb = slice(ls.POSE_START + b * _C, ls.POSE_START + (b + 1) * _C)
        out[:, pa], out[:, pb] = out[:, pb].copy(), out[:, pa].copy()
    return out


def active_span(times: np.ndarray, vecs: np.ndarray) -> tuple[float, float]:
    """Time range of the sign proper: first to last frame with hand motion, plus a small pad.

    Annotated clips often hold an idle lead-in and tail. A frame is active when a hand is
    present and the summed frame-to-frame change of the hand blocks is at least
    ACTIVE_MOTION_REL of the clip's 90th-percentile motion. Without any active frame the whole
    clip is returned.
    """
    times = np.asarray(times, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float32)
    hands = _present(vecs, _LEFT) | _present(vecs, _RIGHT)
    m = np.zeros(len(times))
    for block in (_LEFT, _RIGHT):
        present = _present(vecs, block)
        both = present[1:] & present[:-1]
        change = np.abs(np.diff(vecs[:, block], axis=0)).sum(axis=1)
        m[1:] += np.where(both, change, 0.0)
    ref = float(np.percentile(m[hands], 90)) if hands.any() else 0.0
    active = hands & (m >= ACTIVE_MOTION_REL * ref) if ref > 0 else hands
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return float(times[0]), float(times[-1])
    return (
        float(max(times[0], times[idx[0]] - ACTIVE_PAD_S)),
        float(min(times[-1], times[idx[-1]] + ACTIVE_PAD_S)),
    )


def natural_resample(times: np.ndarray, vecs: np.ndarray, duration: float) -> np.ndarray:
    """Resample [0, duration] to FPS frames per second; never cuts.

    Up to MAX_TAKE_S the sign keeps its natural speed. A longer span gets the maximum
    number of frames spread over the whole span, so it plays a little faster.
    """
    span = float(duration)
    if span <= 0:
        raise ValueError("duration must be positive")
    n = min(int(math.ceil(span * FPS - 1e-9)) + 1, int(round(MAX_TAKE_S * FPS)) + 1)
    return resample_window(
        np.asarray(times), np.asarray(vecs), end_time=span, length=n, span_s=span
    )


def clean_clip(times: np.ndarray, vecs: np.ndarray, duration: float) -> np.ndarray | None:
    """One playable take from a stored MS-ASL sequence, or None if the hands are mostly missing."""
    if duration <= 0:
        return None
    times, vecs = np.asarray(times, dtype=np.float64), np.asarray(vecs, dtype=np.float32)
    keep = (times >= -1e-9) & (times <= duration + 1e-9)
    t, v = times[keep], vecs[keep]
    if len(t) == 0:
        return None
    t0, t1 = active_span(t, v)
    keep = (t >= t0) & (t <= t1)
    t, v = t[keep], v[keep]
    posed = np.flatnonzero(_present(v, _POSE))  # no blank frame at either edge
    if len(posed) == 0:
        return None
    t, v = t[posed[0] : posed[-1] + 1], v[posed[0] : posed[-1] + 1]
    t = t - t[0]
    if hand_coverage(v) < MIN_HAND_COVERAGE:
        return None
    v = fill_gaps(t, v)
    if motion(v, _LEFT) > motion(v, _RIGHT):
        v = mirror(v)
    if len(t) < 2 or t[-1] <= 0:
        return None
    return natural_resample(t, v, t[-1])


def _compare_form(take: np.ndarray) -> np.ndarray:
    n = len(take)
    if n < 2:
        return np.repeat(take, COMPARE_STEPS, axis=0)
    span = (n - 1) / FPS
    return resample_window(
        np.arange(n) / FPS, take, end_time=span, length=COMPARE_STEPS, span_s=span
    )


def rank(takes: list[np.ndarray]) -> list[int]:
    """Indices from most to least typical (smallest summed distance to the others first)."""
    if len(takes) <= 1:
        return list(range(len(takes)))
    flat = np.stack([_compare_form(t) for t in takes]).reshape(len(takes), -1).astype(np.float64)
    sq = (flat**2).sum(axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * flat @ flat.T, 0.0))
    return [int(i) for i in np.argsort(dist.sum(axis=1), kind="stable")]


# word -> [(clip id, take)], most typical first
Candidates = dict[str, list[tuple[str, np.ndarray]]]


def collect_candidates(clips: list, work: Path | str) -> Candidates:
    """Cleaned takes per word from the extracted clips, as (clip id, take), most typical first."""
    from senseless.collect import msasl

    groups: Candidates = {}
    for clip in clips:
        if clip.status != "extracted":
            continue
        path = msasl.sequence_path(Path(work), clip)
        if not path.exists():
            continue
        with np.load(path) as seq:
            take = clean_clip(seq["times"], seq["vecs"], float(seq["duration"]))
        if take is not None:
            groups.setdefault(clip.label, []).append((clip.clip_id, take))
    return {
        w: [items[i] for i in rank([take for _, take in items])]
        for w, items in sorted(groups.items())
    }


def save_candidates(candidates: Candidates, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for word, items in candidates.items():
        arrays[f"{word}__ids"] = np.array([cid for cid, _ in items], dtype=str)
        for k, (_, take) in enumerate(items):
            arrays[f"{word}__{k}"] = take
    np.savez_compressed(path, **arrays)
    return path


def load_candidates(path: Path | str) -> Candidates:
    out: Candidates = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            word, tail = key.rsplit("__", 1)
            if tail != "ids":
                continue
            ids = [str(i) for i in archive[key]]
            out[word] = [(cid, archive[f"{word}__{k}"]) for k, cid in enumerate(ids)]
    return dict(sorted(out.items()))


def choose(candidates: Candidates, choices: dict[str, str]) -> dict[str, np.ndarray]:
    """One take per word: the chosen clip id (default the first, most typical); none if rejected."""
    for word, pick in choices.items():
        if word not in candidates:
            raise ValueError(f"choices name an unknown word: {word}")
        if not isinstance(pick, str) or (
            pick != "reject" and pick not in [cid for cid, _ in candidates[word]]
        ):
            raise ValueError(f"bad choice for {word}: {pick!r}")
    out: dict[str, np.ndarray] = {}
    for word, items in candidates.items():
        pick = choices.get(word)
        if pick == "reject":
            continue
        out[word] = next((t for cid, t in items if cid == pick), items[0][1])
    return out


def _preview(take: np.ndarray) -> list:
    """15 FPS, x/y only, in hundredths: [pose 18, left hand 42 or [], right hand 42 or []]."""
    frames = []
    for vec in take[::2]:
        parts = []
        for block in (_POSE, _LEFT, _RIGHT):
            xy = vec[block].reshape(-1, _C)[:, :2]
            keep = block == _POSE or np.any(vec[block])
            parts.append(np.round(xy * 100).astype(int).ravel().tolist() if keep else [])
        frames.append(parts)
    return frames


def build_review_html(candidates: Candidates) -> str:
    """A self-contained page that plays every word's takes as the app's stick figure."""
    from senseless.common.config import UI
    from senseless.ui.figure import HAND_EDGES, HEAD_RADIUS, POSE_EDGES

    data = {
        "words": [
            {"word": w, "ids": [cid for cid, _ in items], "takes": [_preview(t) for _, t in items]}
            for w, items in candidates.items()
        ],
        "pose_edges": POSE_EDGES,
        "hand_edges": HAND_EDGES,
        "head": HEAD_RADIUS,
        "extent": list(UI.figure_extent),
    }
    return _PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign takes review</title>
<style>
body{font-family:system-ui,sans-serif;background:#13262b;color:#e8eef0;margin:16px}
header{position:sticky;top:0;background:#13262b;padding:8px 0;display:flex;gap:12px;
  align-items:center;z-index:1;flex-wrap:wrap}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.card{background:#1c363d;border-radius:8px;padding:8px}
.card.rejected{opacity:.35}
.card h3{margin:0 0 4px;font-size:15px}
canvas{background:#0e1d21;border-radius:4px;width:100%;height:auto}
.btns button{margin:4px 4px 0 0;min-width:28px}
button.on{outline:2px solid #f0a030}
</style></head><body>
<header><strong>Sign takes review</strong><span id="count"></span>
<button id="export">Export choices</button></header>
<p>Each card loops the automatic pick (1). Click another number to use an alternate, or
Reject to leave the word out. Reject words whose take shows a different sense than the
usual spoken word (e.g. LIKE, RIGHT, WATCH). Then export <code>choices.json</code> and run
<code>python -m senseless.sign.extra_library build --choices choices.json</code>.</p>
<div id="grid"></div>
<script>
const D = __DATA__;
const choice = {};
const [xmin, xmax, ymin, ymax] = D.extent;
const cards = [];
function draw(ctx, w, h, f) {
  ctx.clearRect(0, 0, w, h);
  const [pose, left, right] = f;
  if (!pose.some(v => v !== 0)) return;
  const s = Math.min(w / (xmax - xmin), h / (ymax - ymin));
  const ox = w / 2 - s * (xmin + xmax) / 2, oy = h / 2 - s * (ymin + ymax) / 2;
  const P = (a, i) => [ox + s * Math.min(xmax, Math.max(xmin, a[2 * i] / 100)),
                       oy + s * Math.min(ymax, Math.max(ymin, a[2 * i + 1] / 100))];
  const seg = (a, i, j, col) => { const p = P(a, i), q = P(a, j); ctx.strokeStyle = col;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
  ctx.lineWidth = 2;
  for (const [i, j] of D.pose_edges) seg(pose, i, j, "#dfe7ea");
  const n = P(pose, 0);
  ctx.beginPath(); ctx.arc(n[0], n[1], D.head * s, 0, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 1.5;
  if (left.length) for (const [i, j] of D.hand_edges) seg(left, i, j, "#4cc38a");
  if (right.length) for (const [i, j] of D.hand_edges) seg(right, i, j, "#f0a030");
}
function updateCount() {
  const rejected = Object.values(choice).filter(c => c === "reject").length;
  document.getElementById("count").textContent =
    `${D.words.length} words, ${rejected} rejected`;
}
for (const item of D.words) {
  const card = document.createElement("div"); card.className = "card";
  const title = document.createElement("h3"); title.textContent = item.word;
  const canvas = document.createElement("canvas"); canvas.width = 170; canvas.height = 160;
  const btns = document.createElement("div"); btns.className = "btns";
  const state = { item, canvas, ctx: canvas.getContext("2d"), pick: 0,
                  t0: performance.now(), last: -1 };
  const buttons = [];
  const select = (pick) => {
    choice[item.word] = pick; state.pick = pick === "reject" ? 0 : pick;
    state.t0 = performance.now(); state.last = -1;
    card.classList.toggle("rejected", pick === "reject");
    buttons.forEach((b, k) =>
      b.classList.toggle("on", (k === item.takes.length ? "reject" : k) === pick));
    updateCount();
  };
  item.takes.forEach((_, k) => {
    const b = document.createElement("button"); b.textContent = String(k + 1);
    b.onclick = () => select(k); buttons.push(b); btns.appendChild(b);
  });
  const rej = document.createElement("button"); rej.textContent = "Reject";
  rej.onclick = () => select("reject"); buttons.push(rej); btns.appendChild(rej);
  buttons[0].classList.add("on");
  card.append(title, canvas, btns); document.getElementById("grid").appendChild(card);
  cards.push(state);
}
function tick(now) {
  for (const c of cards) {
    const frames = c.item.takes[c.pick];
    const k = Math.floor((now - c.t0) / 1000 * 15) % (frames.length + 8);
    if (k !== c.last) { c.last = k; draw(c.ctx, 170, 160, frames[Math.min(k, frames.length - 1)]); }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
updateCount();
document.getElementById("export").onclick = () => {
  const out = {};
  for (const item of D.words) {
    const c = choice[item.word];
    if (c === "reject") out[item.word] = "reject";
    else if (c) out[item.word] = item.ids[c];
  }
  const a = document.createElement("a");
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  a.href = URL.createObjectURL(blob);
  a.download = "choices.json"; a.click();
};
</script></body></html>
"""


def main() -> None:
    from senseless.collect import msasl
    from senseless.common.config import DATA_DIR, PATHS
    from senseless.sign.library import add_extra, build_library, save_library

    parser = argparse.ArgumentParser(description="MS-ASL takes for the signing figure.")
    parser.add_argument("step", choices=["review", "build"])
    parser.add_argument("--work", default=str(msasl.ANIM_WORK_DIR))
    parser.add_argument("--choices", default=None, help="choices.json from the review page.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--out",
        default=str(PATHS.sign_library),
        help="The combined library (ours + MS-ASL). The MS-ASL takes always go to "
        "models/sign_library_extra.npz.",
    )
    args = parser.parse_args()
    work = Path(args.work)
    cand_path = work / "candidates.npz"

    if args.step == "review":
        clips = msasl.read_manifest(work / "manifest.csv")
        candidates = collect_candidates(clips, work)
        save_candidates(candidates, cand_path)
        page = work / "review.html"
        page.write_text(build_review_html(candidates), encoding="utf-8")
        n = sum(len(t) for t in candidates.values())
        print(f"{len(candidates)} words, {n} candidate takes -> {page}")
        missing = sorted({c.label for c in clips} - set(candidates))
        if missing:
            print(f"no usable clip for {len(missing)} words: {', '.join(missing)}")
        return

    if not cand_path.exists():
        raise SystemExit(f"run review first: {cand_path} not found")
    candidates = load_candidates(cand_path)
    if not candidates:  # e.g. a file from before clip ids: never wipe the existing takes
        raise SystemExit(f"no candidate takes in {cand_path}; re-run review")
    choices = json.loads(Path(args.choices).read_text(encoding="utf-8")) if args.choices else {}
    if not isinstance(choices, dict):
        raise SystemExit(f"{args.choices} must hold a JSON object (word -> clip id or 'reject')")
    extra = choose(candidates, choices)
    save_library(extra, PATHS.sign_library_extra)
    ours = build_library(args.data_dir)
    library = add_extra(ours, PATHS.sign_library_extra)
    path = save_library(library, args.out)
    print(f"saved {path}: {len(library)} words ({len(library) - len(ours)} from MS-ASL)")


if __name__ == "__main__":
    main()
