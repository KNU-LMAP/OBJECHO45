"""Additive White Gaussian Noise (AWGN) at a target SNR, as used to build the noisy sets."""
import numpy as np

SNR_DB_LEVELS = (30, 20, 10, 0, -5, -10)
SEED = 1

def level_name(snr_db):
    """Directory name for an SNR level: 30 -> '30dB', 0 -> '00dB', -5 -> '-5dB'."""
    return f"{snr_db:02d}dB" if snr_db >= 0 else f"{snr_db}dB"

def awgn(signal, snr_db, rng):
    """Additive White Gaussian Noise to `signal` at a target SNR in dB.
    
    SNR is defined on power: SNR_dB = 10 * log10(P_signal / P_noise), so the noise variance
    needed is P_signal / 10**(SNR_db / 10). Each noise sample is an independent draw from
    N(0, sigma^2) where signma^2 is the noise variance.

    P_signal is measured per clip and per channel, not once over the whole dataset:
    clip power varies by about 91x and the two receivers' power can differ by up to about 10 dB,
    so a single noise level would not give any individual clip or channel the nominal SNR.
    `signal` has shape (samples, channels); a 1-D (mono) array works the same way.

    `rng` is a numpy random number generator. The released sets use on genrator per SNR level,
    created with `np.random.default_rng(SEED)`, so a clip gets the same noise pattern at every
    SNR level and the levels differ only in amplitude.
    """
    signal_power = np.mean(signal**2, axis=0)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise