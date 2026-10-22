"""Build the OBJECHO45 v1.0 release archives from the original recordings.

Writes the zip files and SHA256SUMS to --out. Only the 13,419 clips listed in metadata/clips.csv
are included. Inside each zip the paths are what scripts/download.py unpacks into data/:

  objecho45_clean_4ms.zip                clean_4ms/<clip>.wav
  objecho45_noisy_4ms_<level>.zip        noisy_4ms/<level>/<clip>.wav
  objecho45_noisy_32ms_<level>.zip       noisy_32ms/<level>/<clip>.wav
  objecho45_auditory_<level>_part1.zip   auditory/<level>/<clip>.npy   (clips in folds 1-5)
  objecho45_auditory_<level>_part2.zip   auditory/<level>/<clip>.npy   (clips in folds 6-10)

The auditory spectrograms are converted from float64 to float32. Each level is split in two
because one level is about 2.2 GB and a release file may be at most 2 GiB.

Usage:
  python scripts/make_release.py --research-repo ~/lmap/ultrasonic_echo_discrimination_45classes
  python scripts/make_release.py --research-repo ... --levels 30      # a quick test build
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import zipfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objecho45.noise import SNR_DB_LEVELS, level_name  # noqa: E402

# Folder names inside the research repo
CLEAN = "echo_dataset_new"
NOISY_4MS = "echo_dataset_noised_new"
NOISY_32MS = "echo_dataset_noised_new_32ms"
AUDITORY = "echo_dataset_noised_new_spectrogram"

PARTS = {1: range(1, 6), 2: range(6, 11)}   # which folds go into each auditory part


def zip_files(path, items):
    """Zip files as they are. items: (name inside the zip, file on disk). Audio barely
    compresses, so files are stored without compression."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for arcname, source in tqdm(items, desc=os.path.basename(path)):
            archive.write(source, arcname)


def zip_spectrograms(path, items):
    """Like zip_files, but each .npy array is converted from float64 to float32."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for arcname, source in tqdm(items, desc=os.path.basename(path)):
            buffer = io.BytesIO()
            np.save(buffer, np.load(source).astype(np.float32))
            archive.writestr(arcname, buffer.getvalue())


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--research-repo", required=True, help="folder holding the four source datasets")
    parser.add_argument("--out", default="release")
    parser.add_argument("--metadata", default="metadata")
    parser.add_argument("--levels", type=int, nargs="+", default=list(SNR_DB_LEVELS))
    args = parser.parse_args()

    with open(os.path.join(args.metadata, "clips.csv"), newline="") as f:
        clips = [row["clip_id"] for row in csv.DictReader(f)]
    with open(os.path.join(args.metadata, "folds_10.csv"), newline="") as f:
        fold_of = {row["clip_id"]: int(row["fold"]) for row in csv.DictReader(f)}
    assert len(clips) == 13419 and len(fold_of) == 13419

    src = lambda folder, *parts: os.path.join(args.research_repo, folder, *parts)
    os.makedirs(args.out, exist_ok=True)
    built = []

    name = "objecho45_clean_4ms.zip"
    zip_files(os.path.join(args.out, name),
              [(f"clean_4ms/{c}.wav", src(CLEAN, c + ".wav")) for c in clips])
    built.append(name)

    for snr_db in args.levels:
        level = level_name(snr_db)
        for set_name, folder in (("noisy_4ms", NOISY_4MS), ("noisy_32ms", NOISY_32MS)):
            name = f"objecho45_{set_name}_{level}.zip"
            zip_files(os.path.join(args.out, name),
                      [(f"{set_name}/{level}/{c}.wav", src(folder, level, c + ".wav")) for c in clips])
            built.append(name)
        for part, folds in PARTS.items():
            name = f"objecho45_auditory_{level}_part{part}.zip"
            zip_spectrograms(os.path.join(args.out, name),
                             [(f"auditory/{level}/{c}.npy", src(AUDITORY, level, c + ".npy"))
                              for c in clips if fold_of[c] in folds])
            built.append(name)

    # Checksums, in the format `sha256sum` writes and `sha256sum -c` reads
    with open(os.path.join(args.out, "SHA256SUMS"), "w") as f:
        for name in sorted(built):
            f.write(f"{sha256_of(os.path.join(args.out, name))}  {name}\n")
    print(f"built {len(built)} zip files and SHA256SUMS in {args.out}/")


if __name__ == "__main__":
    main()
