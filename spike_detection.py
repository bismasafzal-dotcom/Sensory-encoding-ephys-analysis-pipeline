"""
spike_detection.py
------------------
Detect spikes in an extracellular voltage trace.

Pipeline: bandpass filter -> robust noise-based threshold -> peak detection
with a refractory window. This mirrors a standard single-unit spike-sorting
front end and works on real traces as well as the simulated one.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(
    voltage: np.ndarray,
    fs: float,
    low_hz: float = 300.0,
    high_hz: float = 5000.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass for the spike band (default 300-5000 Hz)."""
    nyq = fs / 2.0
    high_hz = min(high_hz, 0.99 * nyq)
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, voltage)


def robust_threshold(filtered: np.ndarray, k: float = 4.0) -> float:
    """Quiroga-style noise estimate: k * median(|x|) / 0.6745.

    Using the median absolute deviation makes the threshold robust to the
    spikes themselves, unlike the raw standard deviation.
    """
    sigma = np.median(np.abs(filtered)) / 0.6745
    return k * sigma


def detect_spikes(
    voltage: np.ndarray,
    fs: float,
    threshold_k: float = 4.0,
    refractory_ms: float = 1.5,
) -> dict:
    """Detect spike times from a raw extracellular trace.

    Returns a dict: filtered, threshold, spike_times (s), spike_idx (samples).
    Spikes are detected as negative-going excursions (typical extracellular).
    """
    filtered = bandpass_filter(voltage, fs)
    thr = robust_threshold(filtered, k=threshold_k)
    refractory = int(fs * refractory_ms / 1000.0)

    # Negative-going peaks: find peaks on the inverted signal.
    peaks, _ = find_peaks(-filtered, height=thr, distance=max(1, refractory))

    return {
        "filtered": filtered,
        "threshold": thr,
        "spike_idx": peaks,
        "spike_times": peaks / fs,
    }


def detection_report(true_times: np.ndarray, detected_times: np.ndarray,
                     tol_ms: float = 1.0, fs: float = 20000.0) -> dict:
    """Match detected spikes to ground truth and compute precision/recall.

    Useful as a sanity check on the simulated data; omit when running on
    real recordings where ground truth is unknown.
    """
    tol = tol_ms / 1000.0
    matched_true = np.zeros(true_times.size, dtype=bool)
    tp = 0
    for dt_ in detected_times:
        diffs = np.abs(true_times - dt_)
        j = np.argmin(diffs) if diffs.size else None
        if j is not None and diffs[j] <= tol and not matched_true[j]:
            matched_true[j] = True
            tp += 1
    fp = detected_times.size - tp
    fn = true_times.size - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}
