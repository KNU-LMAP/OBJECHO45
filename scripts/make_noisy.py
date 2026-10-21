""""Rebuild the noisy 4 ms sets from the clean recordings.

For each SNR level this adds white Gaussian noise (objecho45.noise.awgn) to every clean clip and 
write 32-bit float WAV files to <out>/<level>/, reproducing the released noisy_4ms files.

The files used in the paper were made with one random generator per level, seeded with 1, that
walked through all 15,668 original recordings in sorted filename order. OBJECHO45 shares only
13,419 of them, so this script advances the generator over the 2,249 excluded recordings (EXCLUDED_COUNTS).
That gives each of the 13,419 clips the same noise as in the paper's files. Without this step, any clip
that follows an excluded recording in sorted order would get different noise from the paper's files.

Usage:
    python scripts/make_noisy.py                    # all six levels -> data/noisy_4ms/
    python scripts/make_noisy.py --levels 30 -5     # only some levels
    python scripts/make_noisy.py --compare DIR      # write nothing; check the result against the
                                                    # files in DIR/<level>/ (e.g. the released set)

NumPy does not promise identical random streams across version. The released files were checked against
NumPy 2.4.4; if another version gives different noise, the released zips are authoritative.
"""

import argparse
import csv
import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objecho45.noise import SEED, SNR_DB_LEVELS, awgn, level_name

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Original recordings that are not part of OBJECHO45: class -> number of clips (indices 0001..N)
EXCLUDED_COUNTS = {
    "cube_90": 427,
    "sphere_10": 237, "sphere_20": 224, "sphere_30": 219, "sphere_45": 236,
    "sphere_55": 232, "sphere_65": 222, "sphere_75": 222, "sphere_90": 230,
}
CLIP_SHAPE = (1000, 2)  # samples x channels of every native recording
SAMPLE_RATE = 250000
N_ORIGINAL = 15668

def original_sequence(clip_ids):
    """All 15,668 original recordings in sorted order. as (clip_id, is_excluded)."""
    
    excluded = [f"{cls}_0.5m_5_{i:04d}" for cls, n in EXCLUDED_COUNTS.items()
                for i in range(1, n+1)]
    sequence = sorted([(c, False) for c in clip_ids] + [(c, True) for c in excluded])
    assert len(sequence) == N_ORIGINAL, len(sequence)
    return sequence

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="data/clean_4ms", help="folder with the clean .wav files")
    parser.add_argument("--out", default="data/noisy_4ms", help="folder to write <level>/ into")
    parser.add_argument("--metadata", default="metadata", help="folder with clips.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=list(SNR_DB_LEVELS))
    parser.add_argument("--compare", metavar="DIR", help="write nothing; compare with the files in DIR/<level>/")
    args = parser.parse_args()

    with open(os.path.join(args.metadata, "clips.csv"), newline="") as f:
        clip_ids = [row["clip_id"] for row in csv.DictReader(f)]
    sequence = original_sequence(clip_ids)

    # The clean clips are the same at every level, so read them once
    clean = {}
    for clip_id in tqdm(clip_ids, desc="reading clean clips"):
        x, sample_rate = sf.read(os.path.join(args.src, clip_id + ".wav"))
        assert x.shape == CLIP_SHAPE and sample_rate == SAMPLE_RATE, (clip_id, x.shape, sample_rate)
        clean[clip_id] = x

    for snr_db in args.levels:
        name = level_name(snr_db)
        rng = np.random.default_rng(SEED)
        out_dir = os.path.join(args.compare or args.out, name)
        if not args.compare:
            os.makedirs(out_dir, exist_ok=True)

        n_different, worst = 0, 0.0
        for clip_id, is_excluded in tqdm(sequence, desc=name):
            if is_excluded:
                rng.standard_normal(CLIP_SHAPE)     # not released, but its draws still count
                continue
            noisy = awgn(clean[clip_id], snr_db, rng)
            path = os.path.join(out_dir, clip_id + ".wav")
            if args.compare:
                released, _ = sf.read(path, dtype="float32")
                diff = float(np.abs(noisy.astype(np.float32) - released).max())
                n_different += diff > 0
                worst = max(worst, diff)
            else:
                sf.write(path, noisy, SAMPLE_RATE, subtype="FLOAT")

        if args.compare:
            print(f"{name}: {len(clip_ids) - n_different}/{len(clip_ids)} clips identical"
                  f" (largest difference {worst:.2e})")
        else:
            print(f"{name}: {len(clip_ids)} clips written to {out_dir}")

if __name__ == "__main__":
    main()