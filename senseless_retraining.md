# Retraining Senseless with additional recordings

This guide starts from a folder of new recordings and ends with the new model running
on the Pi.

The new folder must have the same layout as `data\`: one folder per word, with
`NNNN.npy` files inside, for example `C:\new_recordings\HELLO\0000.npy`. Replace
`C:\new_recordings` below with your folder's path.

Folder names must match the existing labels exactly, for example `THANKYOU`, not
`THANK-YOU` or `thankyou`. A folder with any other name becomes a new class.

## On the PC (PowerShell, from the project folder)

1. Go to the project folder:

```powershell
cd C:\Senseless
```

2. Back up the current data and models:

```powershell
Copy-Item -Recurse data data_backup_$(Get-Date -Format yyyy-MM-dd)
```

```powershell
Copy-Item -Recurse models models_backup_$(Get-Date -Format yyyy-MM-dd)
```

3. Count the samples per word before merging:

```powershell
.venv\Scripts\python -c "from senseless.collect import dataset; print({l: dataset.count_samples(l) for l in dataset.list_labels()})"
```

4. Merge the new recordings into `data\`.

Don't copy the folders over by hand. The new files restart at `0000.npy`, so they
would overwrite existing samples. This command gives each file the next free
number and rejects any file with the wrong shape:

```powershell
.venv\Scripts\python -c "import numpy as np; from pathlib import Path; from senseless.collect import dataset; src = Path(r'C:\new_recordings'); saved = [dataset.save_window(np.load(f), d.name) for d in sorted(src.iterdir()) if d.is_dir() for f in sorted(d.glob('*.npy'))]; print('merged', len(saved), 'samples')"
```

5. Count again to check the merge:

```powershell
.venv\Scripts\python -c "from senseless.collect import dataset; print({l: dataset.count_samples(l) for l in dataset.list_labels()})"
```

6. Install the training packages. You only need to do this once per PC:

```powershell
.venv\Scripts\python -m pip install tensorflow scikit-learn matplotlib
```

7. Retrain:

```powershell
.venv\Scripts\python -m senseless.notebooks.train_gru
```

Check the printout:

- **Per-label counts:** every word is there with roughly balanced numbers, about 50
  each.
- **Test accuracy and the "simulated 10 FPS" accuracy:** the previous model scored
  99.2% and 98.4%.
  - New data changes the test split, so the numbers are not directly comparable.
  - A big drop still means something is wrong.
- **Confusion matrix:** a new word should not be confused with a similar sign.
  - Watch two-handed signs made in front of the chest.
  - HOW and BOOK are the current weak pair.
- **TFLite accuracy:** it must match the Keras accuracy. If it doesn't, the export
  went wrong.

**With takes recorded on the Pi** (much better recognition on the Pi):

- Record about 10 takes per word on the Pi, including IDLE, with the app stopped:
  `python -m senseless.collect --label WORD --auto --backend lite --parallel --camera opencv`.
- Copy the Pi's `~/senseless/data` to the PC as `data_pi` (keep it out of `data\`).
- Retrain with them:

```powershell
.venv\Scripts\python -m senseless.notebooks.train_gru --pi-data data_pi
```

- Check the extra **Pi test accuracy**. It is measured on 30% of the Pi takes, which
  the model did not train on, and is the best offline estimate of accuracy on the Pi.
- For the model you deploy, train on all the Pi takes:

```powershell
.venv\Scripts\python -m senseless.notebooks.train_gru --pi-data data_pi --pi-test-frac 0
```

8. Rebuild the sign library for the Speech-mode signing figure:

```powershell
.venv\Scripts\python -m senseless.sign.library
```

This keeps the MS-ASL takes for words you did not record (`models/sign_library_extra.npz`). A word you record yourself replaces its MS-ASL take.

People may say a new word differently from its label, for example "thanks" for
THANKYOU. If so, add the spoken form to `UI.sign_aliases` in
`senseless/common/config.py`. That is a code change, so commit it through a PR.

9. Test live in onset mode:

```powershell
.venv\Scripts\python -m senseless.sign.demo
```

Sign each word a few times, new and old, and check that nothing got worse.

10. Test the app:

```powershell
.venv\Scripts\python -m senseless.ui --windowed
```

In Speech mode, say the new words and check that they are highlighted and signed by
the figure.

11. Copy the new model files to the Pi. Run this on the PC, not on the Pi:

```powershell
scp models\sign_gru_int8.tflite models\sign_labels.txt models\sign_library.npz admin@senseless.local:~/senseless/models/
```

## On the Pi (over SSH)

12. Stop the running app:

```bash
pkill -f senseless.ui
```

13. Start it again so it loads the new model:

```bash
DISPLAY=:0 ~/senseless/deploy/senseless-ui.sh &
```

Then sign every word on the touchscreen. The Pi test matters most:

- The PC numbers come from your own recordings.
- The Pi uses the lighter hand model and a lower frame rate (6.5 to 7.5 FPS).

You can also test without the app, using the headless demo:

```bash
cd ~/senseless && .venv/bin/python -m senseless.sign.demo --backend lite --parallel --camera opencv --headless
```

## If the new model is worse: roll back (on the PC)

Replace `<date>` with the date of the backup from step 2.

```powershell
Copy-Item -Recurse -Force models_backup_<date>\* models\
```

To undo the merge as well, restore the data:

```powershell
Remove-Item -Recurse -Force data
```

```powershell
Copy-Item -Recurse data_backup_<date> data
```

Then repeat step 11 to copy the old model back to the Pi, and restart the app
(steps 12 and 13).

## Notes

- `models\` and `data\` are not in git. The backups from step 2 are your only copy
  of the previous state, so keep them until the new model has passed the Pi test.
- Before step 4, look at a few of the new recordings. Takes that were recorded with a
  different form of the sign, a different camera position or no hands visible make
  the model worse. More data does not make up for that.
