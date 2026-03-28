import numpy as np
import pynapple as nap
from scipy import signal


# Mel settings.
MEL_PARAMS = {
    "num_bands": 80,
    "window_sec": 0.025,
    "hop_sec": 0.0125,
    "fmin": 0.0,
    "fmax": 8000.0,
    "n_fft": 2048,
    "passband_hz": 60.0,
    "stopband_hz": 50.0,
    "stopband_attenuation_db": 70.0,
    "passband_ripple_db": 0.1,
}


def highpass_speech(waveform: np.ndarray, sample_rate: float) -> np.ndarray:
    # Apply the LMV speech high-pass filter.
    sos = signal.iirdesign(
        wp=MEL_PARAMS["passband_hz"],
        ws=MEL_PARAMS["stopband_hz"],
        gpass=MEL_PARAMS["passband_ripple_db"],
        gstop=MEL_PARAMS["stopband_attenuation_db"],
        ftype="ellip",
        output="sos",
        fs=sample_rate,
    )
    return signal.sosfiltfilt(sos, waveform)


def compute_mel_spectrogram(tsd: nap.Tsd, source: str) -> nap.TsdFrame:
    # Read timestamps and samples.
    t = tsd.times()
    w = tsd.values
    sample_rate = estimate_sample_rate(t)

    # Build the spectrogram window.
    window = signal.windows.hamming(
        int(round(MEL_PARAMS["window_sec"] * sample_rate)),
        sym=False,
    )
    nperseg = len(window)
    noverlap = nperseg - int(round(MEL_PARAMS["hop_sec"] * sample_rate))

    # Compute the power spectrogram.
    freqs, rel_times, power = signal.spectrogram(
        w,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=MEL_PARAMS["n_fft"],
        detrend=False,
        scaling="spectrum",
        mode="psd",
    )

    # Restrict the frequency range.
    mask = (freqs >= MEL_PARAMS["fmin"]) & (freqs <= MEL_PARAMS["fmax"])
    freqs = freqs[mask]
    power = power[mask]

    # Project onto mel filters.
    mel_filter, mel_freqs = _mel_filterbank(freqs)
    mel_power = mel_filter @ power
    mel_db = 10.0 * np.log10(mel_power + np.finfo(np.float32).eps)
    abs_times = rel_times + t[min(len(t) - 1, nperseg // 2)]

    # Build the mel frame.
    mel_values = mel_db.T
    columns = [f"{source}_{i:02d}" for i in range(mel_values.shape[1])]
    mel = nap.TsdFrame(t=abs_times, d=mel_values, columns=columns)
    mel.set_info(
        {
            "source": [source] * len(columns),
            "mel_bin": np.arange(len(columns)),
            "frequency_hz": mel_freqs,
        }
    )
    return mel


def estimate_sample_rate(t: np.ndarray) -> float:
    # Estimate the sample rate from timestamps
    return 1.0 / np.median(np.diff(t))


def _mel_filterbank(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Compute mel band edges.
    mel_edges = np.linspace(
        _hz_to_mel(MEL_PARAMS["fmin"]),
        _hz_to_mel(MEL_PARAMS["fmax"]),
        MEL_PARAMS["num_bands"] + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    mel_freqs = hz_edges[1:-1]

    # Build triangular filters.
    filters = np.zeros((MEL_PARAMS["num_bands"], len(freqs)), dtype=np.float64)
    for i in range(MEL_PARAMS["num_bands"]):
        left, center, right = hz_edges[i : i + 3]
        rising = (freqs >= left) & (freqs <= center)
        falling = (freqs >= center) & (freqs <= right)
        filters[i, rising] = (freqs[rising] - left) / (center - left)
        filters[i, falling] = (right - freqs[falling]) / (right - center)

    # Normalize band areas.
    filters *= (2.0 / (hz_edges[2:] - hz_edges[:-2]))[:, None]
    return filters, mel_freqs


def _hz_to_mel(freq_hz: float | np.ndarray) -> np.ndarray:
    # Convert hertz to mel.
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(freq_mel: float | np.ndarray) -> np.ndarray:
    # Convert mel to hertz.
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)
