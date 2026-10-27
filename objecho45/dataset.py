"""Raw access to OBJECHO45: clips, class indices and folds, with no preprocessing."""
import csv
import os
import numpy as np 
import soundfile as sf
from .noise import SNR_DB_LEVELS, level_name

# Directory name under the data root --> file extension
VERSIONS = {
        "clean_4ms": ".wav",
        "noisy_4ms": ".wav",
        "noisy_32ms": ".wav",
        "auditory": ".npy",
        }

DEFAULT_METADATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata")

def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

class OBJECHO45:
    """The clips of one version of OBJECHO45.
    
    Args:
        root: folder that holds clean_4ms/, noisy_4ms/, noisy_32ms/ and auditory/
        version: "clean_4ms", "noisy_4ms", "noisy_32ms" or "auditory"
        snd_db: one of 30, 20, 10, 0, -5, -10 for every version except "clean_4ms"
        folds: keep only the clips in these folds (1-10); None keeps all 13,419
        metadata_dir: folder with classes.csv and folds_10.csv

    `ds[i]` returns `(x, class_index)`. Waveforms come back as float32 arrays of shape
    (samples, channels), scaled to [-1, 1] by the file format and otherwise untouched;
    auditory spectrograms come back as stored, shape (128, 320). `ds.classes[class_index]`
    is the class name and `ds.clips[i]` holds the clip id, class, fold and path.

    The paper's cross-validation uses one fold for validation and the other nine for training:

        val = OBJECHO45(root, "noisy_4ms", 30, folds=[k])
        train = OBJECHO45(root, "noisy_4ms", 30, folds=[f for f in range(1, 11) if f != k])
    """

    def __init__(self, root="data", version="noisy_4ms", snr_db=None, folds=None, metadata_dir=None):
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {sorted(VERSIONS)}, got {version!r}")
        if version == "clean_4ms":
            if snr_db is not None:
                raise ValueError("clean_4ms has no SNR level; leave snr_db as None")
        elif snr_db not in SNR_DB_LEVELS:
            raise ValueError(f"snr_db must be one of {SNR_DB_LEVELS}, got {snr_db!r}")

        metadata_dir = metadata_dir or DEFAULT_METADATA_DIR
        class_index = {                             # maps each class name to its numeric index
            r["class"]: int(r["class_index"])
            for r in _read_csv(os.path.join(metadata_dir, "classes.csv"))
            }
        self.classes = sorted(class_index, key=class_index.get)

        folder = os.path.join(root, version)
        if snr_db is not None:
            folder =  os.path.join(folder, level_name(snr_db))
        ext = VERSIONS[version]
        keep = None if folds is None else set(folds)

        self.clips = []
        for row in _read_csv(os.path.join(metadata_dir, "folds_10.csv")):
            fold = int(row["fold"])
            if keep is not None and fold not in keep:
                continue
            clip_id = row["clip_id"]
            obj, angle = clip_id.split("_")[:2]
            self.clips.append({
                "clip_id": clip_id,
                "class": f"{obj}_{angle}",
                "class_index": class_index[f"{obj}_{angle}"],
                "fold": fold,
                "path": os.path.join(folder, clip_id + ext),
            })

        if self.clips and not os.path.exists(self.clips[0]["path"]):
            raise FileNotFoundError(
                f"{self.clips[0]["path"]} not found. Download the data first (see README) "
                f"and check that root={root!r} is the folder that holds {version}/."
            )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        if clip["path"].endswith(".npy"):
            x = np.load(clip["path"])
        else:
            x, _ = sf.read(clip["path"], dtype="float32", always_2d=True)
        return x, clip["class_index"]

