"""MS-ASL pipeline: selection, download, extraction and windows, without network or tracker."""

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
