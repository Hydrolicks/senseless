"""MS-ASL pipeline: selection, download, extraction and windows, without network or tracker."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from senseless.collect import msasl
from senseless.common import landmark_schema as ls
from senseless.sign.landmarks import RawLandmarks


def _clip(**kw) -> msasl.Clip:
    base = dict(
        clip_id="train_00001",
        label="HELLO",
        split="train",
        signer=3,
        url="https://youtu.be/x",
        start=2.0,
        end=4.5,
        fps=30.0,
        box="0.1000 0.2000 0.9000 0.8000",
    )
    base.update(kw)
    return msasl.Clip(**base)


def test_gloss_mapping_uses_labels_aliases_and_single_label_synonym_groups() -> None:
    vocab = ["HELLO", "THANKYOU", "GOODBYE", "WATER"]
    synonyms = [["water", "drink water"], ["hello", "goodbye"], ["unrelated", "other"]]
    m = msasl.gloss_to_label(vocab, synonyms)
    assert m["hello"] == "HELLO" and m["water"] == "WATER"
    assert m["thank you"] == m["thanks"] == "THANKYOU"
    assert m["bye"] == "GOODBYE"
    assert m["drink water"] == "WATER"  # synonym group with exactly one of our labels
    assert "unrelated" not in m


def test_select_keeps_only_vocabulary_clips_with_split_ids_and_box() -> None:
    entry = dict(
        clean_text="hello",
        signer_id=7,
        url="u",
        start_time=1.5,
        end_time=3.0,
        fps=25.0,
        box=[0.1, 0.2, 0.9, 0.8],
    )
    other = dict(entry, clean_text="banana")
    clips = msasl.select_clips({"train": [other, entry], "test": [entry]}, {"hello": "HELLO"})
    assert [c.clip_id for c in clips] == ["train_00001", "test_00000"]
    first = clips[0]
    assert (first.label, first.split, first.signer, first.fps) == ("HELLO", "train", 7, 25.0)
    assert first.box == "0.1000 0.2000 0.9000 0.8000"


def test_manifest_round_trip(tmp_path) -> None:
    clips = [_clip(), _clip(clip_id="test_00002", split="test", status="unavailable", note="gone")]
    path = tmp_path / "manifest.csv"
    msasl.write_manifest(clips, path)
    assert msasl.read_manifest(path) == clips


class FakeRun:
    """Stands in for subprocess.run(yt-dlp ...): writes the -o file on success."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode, self.stderr, self.calls = returncode, stderr, []

    def __call__(self, cmd, capture_output, text, timeout=None):
        self.calls.append(cmd)
        out = Path(cmd[cmd.index("-o") + 1])
        if self.returncode == 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"v" * msasl.MIN_VIDEO_BYTES)
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)


def test_ytdlp_command_downloads_only_the_padded_section(tmp_path) -> None:
    cmd = msasl.ytdlp_command(_clip(start=0.2, end=3.0), tmp_path / "a.mp4", ffmpeg="ff.exe")
    assert cmd[1:3] == ["-m", "yt_dlp"]
    assert cmd[cmd.index("--download-sections") + 1] == "*0.00-3.50"  # start clamped at 0
    assert cmd[cmd.index("--ffmpeg-location") + 1] == "ff.exe"
    assert cmd[-1] == "https://youtu.be/x"


def test_download_marks_ok_unavailable_and_failed(tmp_path) -> None:
    ok, gone, broken = _clip(clip_id="a"), _clip(clip_id="b"), _clip(clip_id="c")
    msasl.download_clips([ok], tmp_path, run=FakeRun(), sleep=lambda s: None)
    msasl.download_clips(
        [gone], tmp_path, run=FakeRun(1, "ERROR: Private video. Sign in"), sleep=lambda s: None
    )
    msasl.download_clips(
        [broken], tmp_path, run=FakeRun(1, "ERROR: HTTP 500"), sleep=lambda s: None
    )
    assert (ok.status, gone.status, broken.status) == ("ok", "unavailable", "failed")
    assert "Private video" in gone.note
    assert msasl.video_path(tmp_path, ok).exists()


def test_download_skips_existing_files_and_respects_the_limit(tmp_path) -> None:
    done = _clip(clip_id="done")
    msasl.video_path(tmp_path, done).parent.mkdir(parents=True)
    msasl.video_path(tmp_path, done).write_bytes(b"v" * msasl.MIN_VIDEO_BYTES)
    rest = [_clip(clip_id=f"n{i}") for i in range(3)]
    run = FakeRun()
    counts = msasl.download_clips([done, *rest], tmp_path, run=run, sleep=lambda s: None, limit=2)
    assert len(run.calls) == 2 and counts["skipped"] == 1 and done.status == "ok"
    assert rest[2].status == "selected"


def test_classify_failure_distinguishes_recoverable_from_permanent() -> None:
    assert msasl.classify_failure("ERROR: Requested format is not available") == "failed"
    assert (
        msasl.classify_failure("ERROR: Video unavailable. This video is not available")
        == "unavailable"
    )
    assert (
        msasl.classify_failure("ERROR: [youtube] 1AyT77LqJzQ: This video is unavailable")
        == "unavailable"
    )


def test_download_fails_on_empty_output_file(tmp_path) -> None:
    empty = _clip(clip_id="empty")

    class EmptyFileRun:
        def __call__(self, cmd, capture_output, text, timeout=None):
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0, "", "")

    msasl.download_clips([empty], tmp_path, run=EmptyFileRun(), sleep=lambda s: None)
    assert empty.status == "failed"


def test_download_fails_on_a_header_only_file(tmp_path) -> None:
    # yt-dlp can exit 0 after writing an MP4 header with no frames (seen as 261-byte files).
    stub = _clip(clip_id="stub")

    class HeaderOnlyRun:
        def __call__(self, cmd, capture_output, text, timeout=None):
            out = Path(cmd[cmd.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"h" * 261)
            return subprocess.CompletedProcess(cmd, 0, "", "")

    msasl.download_clips([stub], tmp_path, run=HeaderOnlyRun(), sleep=lambda s: None)
    assert stub.status == "failed"
    assert "261 bytes" in stub.note


def test_download_replaces_an_existing_header_only_file(tmp_path) -> None:
    stub = _clip(clip_id="stub", status="no_detections", note="unreadable video")
    path = msasl.video_path(tmp_path, stub)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"h" * 261)
    run = FakeRun()
    counts = msasl.download_clips([stub], tmp_path, run=run, sleep=lambda s: None)
    assert len(run.calls) == 1 and counts["ok"] == 1
    assert stub.status == "ok" and path.stat().st_size >= msasl.MIN_VIDEO_BYTES


def test_download_handles_timeout(tmp_path) -> None:
    timed_out = _clip(clip_id="timeout")

    class TimeoutRun:
        def __call__(self, cmd, capture_output, text, timeout=None):
            raise subprocess.TimeoutExpired(cmd, timeout)

    msasl.download_clips(
        [timed_out], tmp_path, run=TimeoutRun(), sleep=lambda s: None, timeout_s=300.0
    )
    assert timed_out.status == "failed"
    assert timed_out.note.startswith("timeout")


def _pose() -> np.ndarray:
    pose = np.full((33, 3), 0.5, dtype=np.float32)
    pose[11], pose[12] = (0.40, 0.50, 0.0), (0.60, 0.50, 0.0)
    return pose


class FakeBackend:
    def __init__(self, hands: bool = True) -> None:
        self.hands, self.shapes = hands, []

    def extract(self, frame, timestamp_ms):
        self.shapes.append(frame.shape)
        hand = np.full((21, 3), 0.45, dtype=np.float32) if self.hands else None
        return RawLandmarks(left_hand=hand, right_hand=None, pose=_pose())

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def test_crop_box_adds_a_margin_and_falls_back_to_the_full_frame() -> None:
    assert msasl.crop_box((100, 200, 3), "0.3 0.3 0.7 0.7") == (24, 76, 48, 152)
    assert msasl.crop_box((100, 200, 3), "0.0 0.0 1.0 1.0") == (0, 100, 0, 200)
    assert msasl.crop_box((100, 200, 3), "0.5 0.5 0.51 0.51") == (0, 100, 0, 200)  # too small


def test_extract_sequence_crops_and_times_frames_from_the_sign_start() -> None:
    frames = [np.zeros((100, 200, 3), np.uint8)] * 10
    backend = FakeBackend()
    times, vecs = msasl.extract_sequence(frames, 10.0, "0.3 0.3 0.7 0.7", backend, offset_s=0.5)
    assert backend.shapes[0] == (52, 104, 3)
    assert np.allclose(times[:3], [-0.5, -0.4, -0.3])
    assert vecs.shape == (10, ls.FEATURE_DIM) and vecs.dtype == np.float32


def test_extract_clips_saves_sequences_and_flags_clips_without_hands(tmp_path) -> None:
    good, empty = _clip(clip_id="g", status="ok"), _clip(clip_id="e", status="ok")
    for clip in (good, empty):
        msasl.video_path(tmp_path, clip).parent.mkdir(parents=True, exist_ok=True)
        msasl.video_path(tmp_path, clip).write_bytes(b"v")
    frames = [np.zeros((60, 80, 3), np.uint8)] * 40

    def reader(path):
        return frames, 10.0

    backends = iter([FakeBackend(hands=True), FakeBackend(hands=False)])
    counts = msasl.extract_clips([good, empty], tmp_path, lambda: next(backends), reader=reader)
    assert (good.status, empty.status) == ("extracted", "no_detections")
    assert counts == {"extracted": 1, "no_detections": 1}
    with np.load(msasl.sequence_path(tmp_path, good)) as seq:
        assert seq["vecs"].shape == (40, ls.FEATURE_DIM)
        assert float(seq["duration"]) == 2.5


def test_extract_clips_survives_a_reader_error_and_marks_the_clip_failed(tmp_path) -> None:
    bad, good = _clip(clip_id="bad", status="ok"), _clip(clip_id="good", status="ok")
    for clip in (bad, good):
        msasl.video_path(tmp_path, clip).parent.mkdir(parents=True, exist_ok=True)
        msasl.video_path(tmp_path, clip).write_bytes(b"v")
    frames = [np.zeros((60, 80, 3), np.uint8)] * 40

    def reader(path):
        if path.stem == "bad":
            raise ValueError("corrupt video")
        return frames, 10.0

    counts = msasl.extract_clips([bad, good], tmp_path, lambda: FakeBackend(), reader=reader)
    assert (bad.status, good.status) == ("failed", "extracted")
    assert bad.note == "extract error: corrupt video"
    assert counts == {"failed": 1, "extracted": 1}
    # a failed clip is not retried automatically
    assert msasl.extract_clips([bad], tmp_path, lambda: FakeBackend(), reader=reader) == {}
    assert bad.status == "failed"


def test_sane_fps_replaces_implausible_frame_rates_with_30() -> None:
    assert msasl._sane_fps(float("nan")) == 30.0
    assert msasl._sane_fps(0.0) == 30.0
    assert msasl._sane_fps(90000.0) == 30.0
    assert msasl._sane_fps(25.0) == 25.0


def _sequence(hand_from: float = 0.3):
    times = np.round(np.arange(-0.5, 2.5, 0.1), 3)
    vecs = np.zeros((len(times), ls.FEATURE_DIM), np.float32)
    vecs[:, ls.POSE_START : ls.POSE_END] = 0.2
    vecs[times >= hand_from, ls.LEFT_HAND_START : ls.LEFT_HAND_END] = 0.5
    return times, vecs


def test_onset_is_the_first_hand_frame_at_or_after_the_sign_start() -> None:
    times, vecs = _sequence(hand_from=0.3)
    assert msasl.onset_time(times, vecs) == 0.3
    times, vecs = _sequence(hand_from=-0.4)  # hands already up: the sign start itself
    assert msasl.onset_time(times, vecs) == 0.0


def test_cut_windows_onset_with_shifts_and_whole_mode() -> None:
    times, vecs = _sequence()
    onset = msasl.cut_windows(times, vecs, 2.0, "onset", shifts=(0.0, -0.15, 0.15))
    assert len(onset) == 3 and all(w.shape == (45, ls.FEATURE_DIM) for w in onset)
    whole = msasl.cut_windows(times, vecs, 2.0, "whole")
    assert len(whole) == 1 and whole[0].shape == (45, ls.FEATURE_DIM)


def test_write_windows_puts_train_copies_and_test_singles_in_split_folders(tmp_path) -> None:
    work, out = tmp_path / "work", tmp_path / "out"
    train = _clip(clip_id="train_00001", status="extracted")
    test = _clip(clip_id="test_00002", split="test", status="extracted")
    for clip in (train, test):
        msasl.sequence_path(work, clip).parent.mkdir(parents=True, exist_ok=True)
        times, vecs = _sequence()
        np.savez_compressed(msasl.sequence_path(work, clip), times=times, vecs=vecs, duration=2.0)
    counts = msasl.write_windows([train, test], work, out, "onset")
    assert sorted(p.name for p in (out / "train" / "HELLO").glob("*.npy")) == [
        "train_00001_0.npy",
        "train_00001_1.npy",
        "train_00001_2.npy",
    ]
    assert [p.name for p in (out / "test" / "HELLO").glob("*.npy")] == ["test_00002_0.npy"]
    assert counts == {"train": 3, "test": 1}


def test_coverage_flags_words_with_too_few_train_clips(tmp_path) -> None:
    clips = [_clip(clip_id=f"train_{i:05d}", status="extracted") for i in range(3)]
    table = msasl.coverage(clips, tmp_path)
    assert "HELLO" in table and "LOW" in table


def test_write_windows_clears_a_clips_old_windows_when_the_mode_changes(tmp_path) -> None:
    work, out = tmp_path / "work", tmp_path / "out"
    clip = _clip(clip_id="train_00001", status="extracted")
    other = _clip(clip_id="train_00009", status="extracted")
    for c in (clip, other):
        msasl.sequence_path(work, c).parent.mkdir(parents=True, exist_ok=True)
        times, vecs = _sequence()
        np.savez_compressed(msasl.sequence_path(work, c), times=times, vecs=vecs, duration=2.0)
    msasl.write_windows([clip, other], work, out, "onset")
    folder = out / "train" / "HELLO"
    assert len(list(folder.glob("train_00001_*.npy"))) == 3
    msasl.write_windows([clip], work, out, "whole")
    assert sorted(p.name for p in folder.glob("train_00001_*.npy")) == ["train_00001_0.npy"]
    assert len(list(folder.glob("train_00009_*.npy"))) == 3  # other clips untouched


def _select_args(monkeypatch, tmp_path, *extra) -> Path:
    work = tmp_path / "work"
    monkeypatch.setattr(msasl, "load_msasl", lambda d: ({"train": [_entry()]}, []))
    monkeypatch.setattr(msasl, "vocabulary", lambda: ["HELLO"])
    monkeypatch.setattr("sys.argv", ["msasl", "select", "--work", str(work), *extra])
    return work / "manifest.csv"


def _entry() -> dict:
    return dict(
        clean_text="hello", signer_id=1, url="u", start_time=1.0, end_time=2.0, box=[0, 0, 1, 1]
    )


def test_select_refuses_to_overwrite_an_existing_manifest_unless_forced(
    tmp_path, monkeypatch
) -> None:
    manifest = _select_args(monkeypatch, tmp_path)
    msasl.write_manifest([_clip(status="extracted")], manifest)
    before = manifest.read_bytes()
    with pytest.raises(SystemExit):
        msasl.main()
    assert manifest.read_bytes() == before

    _select_args(monkeypatch, tmp_path, "--force")
    msasl.main()
    clips = msasl.read_manifest(manifest)
    assert [c.status for c in clips] == ["selected"]
