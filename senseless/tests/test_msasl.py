"""MS-ASL pipeline: selection, download, extraction and windows, without network or tracker."""

import subprocess
from pathlib import Path

from senseless.collect import msasl


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

    def __call__(self, cmd, capture_output, text):
        self.calls.append(cmd)
        out = Path(cmd[cmd.index("-o") + 1])
        if self.returncode == 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"video")
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
    msasl.video_path(tmp_path, done).write_bytes(b"x")
    rest = [_clip(clip_id=f"n{i}") for i in range(3)]
    run = FakeRun()
    counts = msasl.download_clips([done, *rest], tmp_path, run=run, sleep=lambda s: None, limit=2)
    assert len(run.calls) == 2 and counts["skipped"] == 1 and done.status == "ok"
    assert rest[2].status == "selected"
