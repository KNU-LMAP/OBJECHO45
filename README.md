# OBJECHO45

Airborne ultrasonic echo dataset and cross-SNR benchmark for object shape discrimination:
13,419 stereo echo recordings (250 kHz, broadband FM pulse) of six rigid shapes at nine
orientations, forming 45 classes.

See [DATASHEET.md](DATASHEET.md) for how the data was recorded and processed.

> **Status:** the dataset is available as release `v1.0` of this repository.
> Baseline code and pretrained weights for the paper's models will follow.

## Download

The data is distributed as assetsof the [`v1.0` release](https://github.com/KNU-LMAP/OBJECHO45/releases/tag/v1.0): one zip per SNR level for each version of the data, 25 zips in total (15.2 GB).
`scripts/download.py` fetches the zips you choose and check them against `metadata/SHA256SUMS` and unpacks them into `/data`. See [DATASHEET.md](DATASHEET.md) for what each version contains.

```bash
git clone https://github.com/KNU-LMAP/OBJECHO45.git
cd OBJECHO45
pip install -r requirements.txt

python scripts/download.py                                   # clean_4m and noisy_4ms, all six SNR levels (0.7 GB)
python scripts/download.py --sets noisy_4ms --levels 30 -5   # chosen sets and levels only
python scripts/downlaod.py --sets noisy_32 auditory          # 32 ms set and auditory spectrograms, all six SNR levels (14.5 GB)
```

SNR levels are 30, 20, 10, 0, -5, -30 db. Rerunning the script skips files that are already unpacked.

## Loading

```Python
from objecho45.dataset import OBJECHO45

ds = OBJECHO45("data", "noisy_4ms", snr_db=30)  # all 13,419 clips; if folds=[...], select specified folds only
x, y = ds[0]                                    # x: float32 (1000, 2), y: class index 0-44
print(len(ds), x.shape, ds.classes[y])
```

## Citation

If you use OBJECHO45, please cite:

```bibtex
@INPROCEEDINGS{chenda-park2026objecho45,
    title       = {{OBJECHO45}: Airborne Ultrasonic Echo Dataset and Cross-{SNR} Benchmark for Discrimination},
    author      = {Chenda, Manith and Park, Sangwook},
    booktitle   = {Proceedings of the IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia)},
    year        = {2026}
}
```

The DOI and URL will be added once published.

## License
Code: MIT. Data: CC BY 4.0. (c) 2026 Laboratory for Machine Auditory Perception (L-MAP), Kangwon National University.