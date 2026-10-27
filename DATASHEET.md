# OBJECHO45 datasheet

## Overview

OBJECHO45 is a dataset of airborne ultrasonic echoes from six rigid 3D shapes, with a cross-SNR benchmark for object-orientation discrimination. It has **13,419 streo clips** in **45 classes**, a fixed 10-fold split, and noisy copies at six SNR levels. For the paper and baseline results, see [README](README.md).

## Recording setup

- **Emitter:** a central ultrasonic speaker playing a synthetic broadband FM pulse (two FM sweeps over 2ms: 50->25 kHz, 100->50 kHz)
- **Receivers:** two ultrasonic microphones, offset 5 cm on each side of the speaker, giving a stereo recording.
- **Target:** one object placed at 50 cm in front of the speaker and microphones; each object is bounded within a 5 x 5 x 5 cm volume.
- **Native recordings:** 2 channels, 250 kHz, 1000 samples (≈ 4 ms), 16-bit PCM.

## Objects, orientations, and classes

| Object (`file name prefix`) | Shape | Classes | Clips |
|---|---|:---:|---:|
| `sphere` | sphere | 1 | 395 |
| `cube` | cube | 8 | 2,791 |
| `cylinder` | cylinder | 9 | 3,305 |
| `octahedron` | cylinder | 9 | 3,086 |
| `spherecone` | hemisphere -> cone (sphere-cone) | 9 | 2,896 |
| `conesphere` | cone -> hemisphere (cone-sphere) | 9 | 946 |

Orientations are `00, 10, 20, 30, 45, 55, 65, 75, 90` degrees. A class is an **object-orientation pair**, such as `cube_45`; `metadata/classes.csv` lists all 45 with their index (0-44, in alphabetical order) and clip count.

**Excluded orientation:** two sysmetries make some object-orientation labels physically indistinguishable, so they aren't part of this dataset:

- `sphere` is rotationally symmetric, so only `sphere_00` is incorporated.
- `cube` has four-fold symmetry about the vertical axis, so `cube_90` shows the same face as `cube_00` and is dropped.

**Class imbalance:** classes range from 58 (`conesphere_20`) to 623 (`cylinder_10`) clips, a 10.7:1 ratio. Thus, reporting macro-averaged accuracy next to pooled accuracy is encouraged.

## Naming
```
<objec>_<angle>_<distance>_<size>_<index>.wav      e.g. cylinder_45_0.5m_5_0168.wav
```

`distance` is always `0.5m` and `size` is always `5` (cm), so neither is a variable. `index` runs from 0001 within each object-orientation group.

## Versions of the data

| Set | Channles | Length | Samples | Format | Notes |
|---|:---:|:---:|:---:|---|---|
| `clean_4ms` | stereo | 4 ms | 1000 | 16-bit PCM | the native recordings |
| `noisy_4ms/<level>` | stereo | 4 ms | 1000 | 32-bit float | clean + white Gaussian noise |
| `noisy_32ms/<level>` | mono | 32 ms | 8000 | 16-bit PCM | channel 1 only; see below |
| `auditory/<level>` | - | - | 128 x 320 | float32 `.npy` | BioNet-A frontend output |

Levels are `30dB, 20dB, 10dB, 00dB, -5dB, -10dB`. All sets use the same clip names.

**Noise:** White Gaussian noise is scaled so that each clip and each channel separately reaches the target SNR. Each clip uses the same noise pattern at every SNR level, so the levels differ only in amplitude. scripts/make_noisy.py rebuilds the 4 ms set exactly from the clean recordings, using one random generator per level (seed 1). That generator originally ran over all 15,668 recordings in sorted order, so the script also advances it over the 2,249 excluded ones. This was checked with NumPy 2.4.4. NumPy does not promise identical random streams across versions, so the files in the release are the reference. The 4 ms copies are stored as float because 16-bit quantization would realize only about 29.6 dB at a 30 dB target.

**32 ms mono:** Each 4 ms echo (channel 1 only) is embedded at samples 1000-1999 of an 8000-sample window, and noise is then added to the whole window. The echo fills 12.5% of the window and the rest carries noise, not silence. These files are 16-bit PCM, so at high SNR the realized SNR is slightly below nominal.

**Auditory spectrograms:** computed from the 32 ms mono clips by the auditory frontend of BioNet-A (Park & Elhilali, 2026). They are stored as raw values, converted from float64 to float32. BioNet5-A's per-clip min-max scaling to [0, 1] is applied by the model, not stored in the files.

## Benchmark protocol

10-fold stratified cross-validation using `metadata/folds_10.csv`. Models train on the 30 dB clips only and are evaluated on the same held-out clips at every SNR. Report pooled and macro accuracy; chance is 1/45 = 0.022. Full protocol and results: see the paper.

## Known limitations

- One synthetic emitter, one recording campaign, and fixed distance (50 cm) and object size.
- Strong class imbalance (10.7:1).
- The 32 ms and auditory sets have one channel, so the level difference between the two channels, a potential cue to orientation, is not available in them.
- Noise is white and Gaussian. Real-world interference (wind, other ultrasonic sources, reverberation) is not covered.

## Personal data and ethics

The recoridngs are ultrasonic echoes from inanimated objectss in a laboratory setting. They contain no speech, no people and no personal or sensitive information.

## License and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). Please cite the paper; see [README](README.md) and `CITATION.cff`.
