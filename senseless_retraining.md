# Retraining Senseless with additional recordings

This guide starts from new recordings and ends with the new model running on the Pi.

The model learns from two folders of recordings, kept separate on the PC:

| Folder | Recorded on | Takes per word | Role |
|---|---|---|---|
| `data\` | the PC, several signers | about 50 | Defines the words; most of the training data |
| `data_pi\` | the Pi itself | about 10 | Adapts the model to the Pi's tracker, frame rate and camera |

Both folders have one folder per word with `NNNN.npy` files inside, for example
`data\HELLO\0000.npy`. Folder names are the labels and must match exactly, for example
`THANKYOU`, not `THANK-YOU` or `thankyou`.

- **A new word** needs both: about 50 takes in `data\` and about 10 in `data_pi\`. The
  trainer skips Pi takes of words that are not in `data\`.
- **IDLE** needs Pi takes too: hands moving naturally in view, not signing.

## Recording

### On the PC (into `data\`)

Record straight into `data\` with the hands-free recorder. Hide both hands between
takes; the next take starts when a hand appears:

```powershell
.venv\Scripts\python -m senseless.collect --label WATER --samples 50 --auto
```

- BACKSPACE deletes the last take if you fumbled it.
- SPACE pauses and resumes. q or Esc quits.
- Use one form of the sign for every take (see `docs/vocab_reference.md`).
- Several signers are better than one.

If someone recorded on another PC, merge their folder in step 3 instead.

### On the Pi (into the Pi's `~/senseless/data`)

Stop the app first, because it holds the camera:

```bash
pkill -f senseless.ui
```

Then record about 10 takes per word:

```bash
cd ~/senseless && DISPLAY=:0 .venv/bin/python -m senseless.collect --label WATER --samples 10 --auto --backend lite --parallel --camera opencv
```

The Pi's `~/senseless/data` keeps every take ever recorded on the Pi. It is the master
copy of the Pi takes, and step 3 copies all of it to the PC.

Restart the app when you are done:

```bash
DISPLAY=:0 ~/senseless/deploy/senseless-ui.sh &
```

## Training (on the PC, PowerShell, from the project folder)

1. Go to the project folder:

```powershell
cd "C:\Users\Asaf Amrani\Desktop\EE Engineering\Fourth Year\Senseless"
```

2. Back up the data and the models:

```powershell
Copy-Item -Recurse data data_backup_$(Get-Date -Format yyyy-MM-dd)
```

```powershell
Copy-Item -Recurse data_pi data_pi_backup_$(Get-Date -Format yyyy-MM-dd)
```

```powershell
Copy-Item -Recurse models models_backup_$(Get-Date -Format yyyy-MM-dd)
```

3. Bring in the new recordings.

**Pi takes:** replace `data_pi\` with a fresh copy of the Pi's folder. Delete the old
copy first; otherwise scp nests the new one inside it as `data_pi\data`:

```powershell
Remove-Item -Recurse -Force data_pi
```

```powershell
scp -r admin@senseless.local:~/senseless/data data_pi
```

**PC takes recorded on another PC:** merge them into `data\`. Don't copy the folders over
by hand: the new files restart at `0000.npy` and would overwrite existing takes. This
command gives each file the next free number and rejects files with the wrong shape.
Replace `C:\new_recordings` with the folder you were given:

```powershell
.venv\Scripts\python -c "import numpy as np; from pathlib import Path; from senseless.collect import dataset; src = Path(r'C:\new_recordings'); saved = [dataset.save_window(np.load(f), d.name) for d in sorted(src.iterdir()) if d.is_dir() for f in sorted(d.glob('*.npy'))]; print('merged', len(saved), 'samples')"
```

Takes you recorded on this PC with the recorder are already in `data\`.

4. Count the takes per word in both folders:

```powershell
.venv\Scripts\python -c "from senseless.collect import dataset; print('PC', {l: dataset.count_samples(l) for l in dataset.list_labels()}); print('Pi', {l: dataset.count_samples(l, 'data_pi') for l in dataset.list_labels('data_pi')})"
```

Every word should have about 50 PC takes and about 10 Pi takes, IDLE included.

5. Install the training packages. You only need to do this once per PC:

```powershell
.venv\Scripts\python -m pip install tensorflow scikit-learn matplotlib
```

6. **Measuring run.** Train with 30% of each word's Pi takes held out, to see how well
   the recipe works on the Pi:

```powershell
.venv\Scripts\python -m senseless.notebooks.train_gru --pi-data data_pi
```

Check the printout:

- **Per-label counts:** every word is there with roughly balanced numbers, about 50
  each.
- **`Pi data:` lines:**
  - the number of Pi takes used and held out;
  - any skipped words, which are in `data_pi\` but not in `data\`.
- **Test accuracy and the "simulated 10 FPS" accuracy,** on the PC test samples. The
  current model scored 98.7% (Keras), and its INT8 file 98.5% and 97.3% at 10 FPS.
  - New data changes the test split, so the numbers are not directly comparable.
  - A big drop still means something is wrong.
- **Pi test accuracy, per word:** this is the best offline estimate of accuracy on the
  Pi.
  - Training runs with Pi takes held out scored about 90% on average.
  - Each word has only 3 held-out takes, so one wrong guess costs a lot.
  - Look for words that fail repeatedly, and for what they are mistaken for.
- **Confusion matrix:** a new word should not be confused with a similar sign.
- **TFLite accuracy:** it must match the Keras accuracy. If it doesn't, the export
  went wrong.

This run overwrites `models\sign_gru_int8.tflite`, but it is not the model to deploy.

7. **Deploy run.** Train the model you will ship on all the Pi takes:

```powershell
.venv\Scripts\python -m senseless.notebooks.train_gru --pi-data data_pi --pi-test-frac 0
```

There is no Pi test accuracy in this printout, because every Pi take is used for
training. Check that the PC test accuracy and the TFLite accuracy look like step 6.

8. Rebuild the sign library for the Speech-mode signing figure:

```powershell
.venv\Scripts\python -m senseless.sign.library
```

This keeps the MS-ASL takes for words you did not record
(`models/sign_library_extra.npz`). A word you record yourself replaces its MS-ASL take.

People may say a new word differently from its label, for example "thanks" for
THANKYOU. If so, add the spoken form to `UI.sign_aliases` in
`senseless/common/config.py`. That is a code change, so commit it through a PR.

9. Test live on the PC in onset mode:

```powershell
.venv\Scripts\python -m senseless.sign.demo
```

Sign each word a few times, new and old, and check that nothing got worse. The PC uses
the full hand model at 30 FPS, so this only catches big mistakes; the Pi test in step 12
matters most.

10. Test the app's Speech mode:

```powershell
.venv\Scripts\python -m senseless.ui --windowed --mode speech
```

Say the new words and check that they are highlighted and signed by the figure.

11. Copy the new model files to the Pi. Run this on the PC, not on the Pi:

```powershell
scp models\sign_gru_int8.tflite models\sign_labels.txt models\sign_library.npz models\sign_library_extra.npz admin@senseless.local:~/senseless/models/
```

## On the Pi (over SSH)

12. Update the code if it changed, then restart the app so it loads the new model:

```bash
cd ~/senseless && git pull && pkill -f senseless.ui
```

```bash
DISPLAY=:0 ~/senseless/deploy/senseless-ui.sh &
```

Then sign every word on the touchscreen, new and old. You can also test without the app,
using the headless demo:

```bash
cd ~/senseless && .venv/bin/python -m senseless.sign.demo --backend lite --parallel --camera opencv --headless
```

## If the new model is worse: roll back (on the PC)

Replace `<date>` with the date of the backups from step 2.

```powershell
Copy-Item -Recurse -Force models_backup_<date>\* models\
```

To undo the data changes as well:

```powershell
Remove-Item -Recurse -Force data, data_pi
```

```powershell
Copy-Item -Recurse data_backup_<date> data
```

```powershell
Copy-Item -Recurse data_pi_backup_<date> data_pi
```

Then repeat step 11 to copy the old model back to the Pi, and restart the app (step 12).

## Notes

- `data\`, `data_pi\` and `models\` are not in git. The backups from step 2 are your only
  copy of the previous state, so keep them until the new model has passed the Pi test.
- The Pi's `~/senseless/data` is the master copy of the Pi takes; back it up too.
- Look at a few new takes before training. Takes recorded with a different form of the
  sign, a different camera position or no hands visible make the model worse. More data
  does not make up for that.
- If a word does badly on the Pi, record 5 to 10 more Pi takes of it. In our tests this
  helped far more than any change to the training augmentation.
