"""Data pipeline of the ICCE-Asia 2026 baselines

Self-contained on purpose: it copies the loading code instead of importing the OBJECHO45
package, so later changes to the package cannot alter the paper's numbers.
"""

import csv
import os
import numpy as np
import soundfile as sf

K_FOLDS = 10
TRAIN_SNR = 30
EVAL_SNRS = [30, 20, 10, 0, -5, -10]
SHAPES = {"4ms": (2, 1000), "32ms": (1, 8000)}      # (channels, samples) of one clip

# Waveform models divide every sample by one scalar: the largest |sample| over the 13,419 clips
# at 30 dB. It is computed once on the training level and then frozen, so a noisier level
# cannot rescale itself by its own noise peak. It is not per clip, because echo amplitude is
# class information.
GLOBAL_SCALE = {"4ms": 0.05893521010875702, "32ms": 0.05889892578125}

METADATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "metadata")

def level_name(snr_db):
    """Directory name for an SNR level: 30 -> '30dB', 0 -> '00dB', -5 -> '-5dB'."""
    return f"{snr_db:02d}dB" if snr_db >= 0 else f"{snr_db}dB"

def load_index(metadata_dir=METADATA_DIR):
    """Clip ids, class indices and fold numbers of the 13,419 benchmark clips, plus the class list.

    The class order is the one the released weights were trained with (classes.csv, indices 0-44),
    so a saved checkpoint's output unit k always means classes[k].
    """

    with open(os.path.join(metadata_dir, "classes.csv"), newline="") as f:
        rows = list(csv.DictReader(f))     # reads the file into a list of dicts, one per data row;
                                           # first line in the CSV becomes keys (e.g. "class": "conesphere_00")
    class_index = {r["class"]: int(r["class_index"]) for r in rows} # maps each class name to its number
                                                                    # e.g. {"conesphere_00": 0, "conesphere_10": 1, ...}
    classes = sorted(class_index, key=class_index.get) # sorts classes by their indices

    with open(os.path.join(metadata_dir, "folds_10.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    clip_ids = [r["clip_id"] for r in rows]     # builds a list of 13,419 filenames
    folds = np.array([int(r["fold"]) for r in rows])    # converts a fold number associated with a clip ID as integer
                                                        # and places them in a NumPy array
    # Retrieve class number by matching cropped clip IDs (e.g. "cube_10_0.5m_5_0221" -> "cube_10")
    # to class indices in the classes.csv
    targets = np.array([class_index["_".join(c.split("_")[:2])] for c in clip_ids])  # c.split("_")[:2] splits by "_" and then keeps only first 2 strings
    return clip_ids, targets, folds, classes

# Loading one SNR level
def load_level(root, version, snr_db, clip_ids):
    """One SNR level as a (N, channels, samples) float32 array in `clip_ids` order, scaled.

    `version` is "4ms" (stereo, 1000 samples) or "32ms" (mono, 8000 samples). Every level is divided by
    the same frozen GLOBAL_SCALE.
    """
    folder = os.path.join(root, f"noisy_{version}", level_name(snr_db))
    arrays = []
    for clip_id in clip_ids:
        audio, _ = sf.read(os.path.join(folder, clip_id + ".wav"), dtype="float32", always_2d=True)  # always_2d=True forces (8000,) -> (8000,1)
        audio = audio.T
        assert audio.shape == SHAPES[version], f"unexpected shape {audio.shape} in {clip_id}"
        arrays.append(audio)
    return np.stack(arrays) / np.float32(GLOBAL_SCALE[version])

# Splitting a fold
def fold_split(folds, k):
    """Training and validation indices for fold k (1-10): validate on fold k, train the rest."""
    val_idx = np.where(folds == k)[0]       # [0] takes the array out of the tuple
    train_idx = np.where(folds != k)[0]
    return train_idx, val_idx
