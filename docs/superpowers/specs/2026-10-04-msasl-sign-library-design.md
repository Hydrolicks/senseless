# MS-ASL takes for the signing figure: design

Date: 2026-10-04. Status: approved in brainstorming.

## Goal

The Speech-mode figure only signs words that are in `models/sign_library.npz`. Today
those are the 49 words we recorded ourselves. This design adds stick-figure takes for
the more common ASL words we did not record, built from MS-ASL landmark data, so more
of a spoken sentence is signed.

MS-ASL is used **only** for these animations. The sign classifier, its training data and
the deployed model do not change. The recognition experiment on the `msasl-experiment`
branch is dropped; only its downloader and extractor are reused here.

## Words

The words are the first 200 glosses of `MSASL_classes.json`, which MS-ASL orders by
frequency. The following are dropped:

- glosses that map to one of our labels (`gloss_to_label`, including MS-ASL synonyms
  and our aliases), because our own takes stay for those words;
- glosses with a space or any character outside `a-z` (`not know`, `not like`,
  `how_many`);
- `hoddog`, a typo in MS-ASL.

That leaves 151 words. The library key of each word is its gloss in upper case, for
example `TEACHER`.

## Pipeline

1. **`msasl select-anim`:**
   - For each word, take up to 5 MS-ASL entries, preferring different signers, in
     split order.
   - Write `C:\Senseless_anim\manifest.csv`, a separate work folder from the
     abandoned experiment.
2. **`msasl download` and `msasl extract`** with `--work C:\Senseless_anim`.
   - These steps are unchanged. Download uses `--pause 5`.
   - Extraction uses the Pi's lite tracker, through `.venv-msasl`.
3. **`python -m senseless.sign.extra_library review`:**
   - For each word, cleans every extracted clip (see below), discards the poor ones and
     ranks the rest.
   - Writes `C:\Senseless_anim\candidates.npz` and a self-contained
     `C:\Senseless_anim\review.html`.
4. **The user reviews the page.**
   - The page plays every word's chosen take and its alternates as the stick figure.
   - The user can reject a word or pick an alternate, then export `choices.json`.
5. **`python -m senseless.sign.extra_library build [--choices choices.json]`:**
   - Writes `models/sign_library.npz` with our own 49 takes (from `data/`, as
     `sign.library` does now) plus the chosen MS-ASL takes.
   - A word with no choice keeps the automatic pick.
   - A word marked `reject` is left out.

## Cleaning one clip

The input is a stored sequence: `times` in seconds relative to the sign start, `vecs`
of shape (N, 153) holding normalized features, and `duration`.

1. **Trim:** keep the frames with `0 <= t <= duration`.
2. **Coverage:** count the share of frames with at least one hand present. A clip
   below 0.6 is discarded.
3. **Gap filling:** a hand block that is missing for at most 0.3 s, with the same hand
   present on both sides, is linearly interpolated per coordinate. A whole frame with
   no pose (an all-zero vector) is filled the same way. Gaps at the start or end stay
   empty.
4. **Dominant hand:** compare the motion energy of each hand block (the summed absolute
   frame-to-frame change). If the left block moves more, mirror the take:
   - negate every x coordinate (features are relative to the shoulder midpoint);
   - swap the two hand blocks;
   - swap the left and right pose points (shoulders, elbows, wrists, hips). The nose
     stays in place.

   Most of our own takes move the right-hand block most (35 of 49), so mirrored MS-ASL
   takes look like ours. A missing (all-zero) point stays zero.
5. **Natural speed:** resample the take to 30 FPS over its own duration, at
   `ceil(duration * 30) + 1` frames, capped at 3.0 s (91 frames). Missing hands are
   held at the nearest frame, using `resample_window`'s rules.

## Ranking

Each word's surviving clips are resampled to 45 steps for comparison only. The
library's `medoid_index` picks the most typical clip as the automatic choice. The rest
are alternates, ordered by their total distance to the other clips. A word with no
surviving clip gets no take.

## Playback

Library takes may now have any number of frames. In `app.py`, `_play_tick_body` passes
the playing take's length to `frame_index`. Our 45-frame takes play exactly as before,
and an MS-ASL take plays at its natural speed of up to 3 s.

## Review page

`review.html` is a single file with no network access and the takes embedded as JSON.

- **Layout:** one card per word, playing the chosen take in a loop on a canvas. It uses
  the same figure as the app: the shoulder, arm and hip lines, a head circle, and hands
  in green and amber.
- **Alternates:** each card has small buttons for its alternates and for "reject".
- **Export:** an "Export choices" button downloads `choices.json`, which maps a word to
  a candidate index or to `"reject"`. Words the user did not change are left out of the
  file.

## Error handling

- Clips that are unavailable or fail to download or extract are skipped, as in the
  experiment.
- A word whose clips all fail is reported by `review` and left out of the library.
- `build` refuses a `choices.json` that names an unknown word, or an index outside that
  word's candidates.

## Testing

Pure functions are tested with synthetic sequences, with no network, tracker or GUI:

- word selection: exclusions, the cap of 5 clips per word and signer diversity;
- trimming, coverage, gap filling (interior gaps filled, edge gaps kept, gaps that are
  too long kept) and mirroring (including mirroring twice to get the original back);
- natural-speed resampling (frame count, cap) and ranking;
- choices handling and library merging, where our own words always win;
- the review page builder (it contains every word and valid JSON);
- the app playing a 90-frame take for 3 s.

## Licence

MS-ASL is used under C-UDA, which covers computational use. The app shows only
landmark skeletons derived from the videos, with no video and no faces. The project
book credits MS-ASL for these takes.

## Out of scope

- multi-word glosses and MS-ASL synonyms as extra spoken words;
- any change to recognition;
- deleting the old experiment's files, which the user will do by hand.
