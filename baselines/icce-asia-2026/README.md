# ICCE-Asia 2026 baselines

Code behind the results table of *OBJECHO45: Airborne Ultrasonic Echo Dataset and Cross-SNR Benchmark for Discrimination* (ICCE-Asia 2026). This directory is frozen and self-contained; it does not import the `objecho45` package, so future changes to the package do not alter the paper's numbers.

**Status: under construction.** The data pipeline is here. The model, training and evaluation code, and the trained weights are coming in subsequent pull requests.

## Models

ConvNeXt1D and ConvNeXt1D+BiGRU, each on the 4 ms stereo and 32 ms mono clips, and BioNet5-A.

## Environment

Python 3.14.6. The exact package versions are pinned:

```bash
pip install -r requirements.txt
```

## What is here

- `requirements.txt`: pinned packages
- `dataset.py`: reads the clips, labels and folds, and applies the paper's scaling
- `utils.py`: class weights for the loss (rare classes count more) and pooled/macro accuracy.
