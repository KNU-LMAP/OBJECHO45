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
        "noisey_32ms": ".wav",
        "auditory": ".npy",
        }

DEFAULT_METADATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata")

def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

class OBJECHO45:
    """This clips of one version of OBJECHO45.  """
