"""Download the OBJECHO45 release files, check them against SHA256SUMS and unpack them.

Each release zip holds paths such as noisy_4ms/30dB/<clip>.wav and is unpacked into --out
(data/ by default), which gives the folder layout that objecho45/dataset.py expects.

Usage:
    python scripts/download.py                                      # the 4 ms set and its 6 SNR levels  (~0.7 GB)
    python scripts/download.py --sets noisy_4ms --levels 30 -10     # specified sets only
    python scripts/download.py --sets noisy_32ms auditory           # get all the 6 SNR levels of 32 ms set and
                                                                    # the auditory spectrogram (~13 GB)

Note:
- GitHub allows < 2 GB per file attached to a release. Thus, the whole auditory spectrograms are divided into 2 parts.
  The script downloads and unpacks both parts in the same folder automatically.
- Files that were downloaded and unpacked before are skipped unless --force is given.
"""

import argparse
import hashlib
import os
import sys
import zipfile
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objecho45.noise import SNR_DB_LEVELS, level_name

BASE_URL = "https://github.com/KNU-LMAP/OBJECHO45/releases/download/v1.0"
SETS = ("clean_4ms", "noisy_4ms", "noisy_32ms", "auditory")

def asset_names(sets, levels):
    """File names of the release zips needed for the chosen sets and SNR levels."""
    names = []
    for name in sets:
        if name == "clean_4ms":
            names.append("objecho45_clean_4ms.zip")
            continue
        for snr_db in levels:
            level = level_name(snr_db)
            if name == "auditory":
                names += [f"objecho45_auditory_{level}_part{p}.zip" for p in (1, 2)]    # each level is split into 2 parts
            else:
                names.append(f"objecho45_{name}_{level}.zip")
    return names


def read_checksums(path):
    """Read a sha256sum-style file: '<hex digest> <file name>' on each line."""
    checksums = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                digest, name = line.split(None, 1)
                checksums[name.strip().lstrip("*")] = digest
    return checksums

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def download(url, path):
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=os.path.basename(path)) as bar:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", nargs="+", choices=SETS, default=["clean_4ms", "noisy_4ms"])
    parser.add_argument("--levels", type=int, nargs="+", default=list(SNR_DB_LEVELS))
    parser.add_argument("--out", default="data", help="folder to unpack into")
    parser.add_argument("--checksums", default="metadata/SHA256SUMS")
    parser.add_argument("--base-url", default=BASE_URL, help="where the release files are")
    parser.add_argument("--keep-zips", action="store_true", help="keep the zips after unpacking")
    parser.add_argument("--force", action="store_true", help="download again even if done before")
    args = parser.parse_args()

    checksums = read_checksums(args.checksums)
    names = asset_names(args.sets, args.levels)
    unlisted = [n for n in names if n not in checksums]
    if unlisted:
        sys.exit("No checksum is listed for: " + ", ".join(unlisted))

    zip_dir = os.path.join(args.out, ".downloads")
    done_dir = os.path.join(args.out, ".done")
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    for name in names:
        marker = os.path.join(done_dir, name + ".ok")
        if os.path.exists(marker) and not args.force:
            print(f"skipping {name} (already downloaded; use --force to repeat)")
            continue

        zip_path = os.path.join(zip_dir, name)
        download(f"{args.base_url}/{name}", zip_path)
        if sha256_of(zip_path) != checksums[name]:
            os.remove(zip_path)
            sys.exit(f"Checksum mismatch for {name}: the download is corrupt. Please try again.")

        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(args.out)
        open(marker, "w").close()                    # remember that this file is complete
        if not args.keep_zips:
            os.remove(zip_path)

    print(f"done: {len(names)} file(s) in {args.out}/")

if __name__ == "__main__":
    main()