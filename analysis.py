"""
analysis.py
-----------
Quantify how the sensory stimulus is encoded in the spike train.

Includes:
  * peri-stimulus time histogram (PSTH) / firing rate
  * cycle-folded phase histogram (phase tuning)
  * vector strength + Rayleigh test for phase locking
  * inter-spike interval (ISI) statistics

These are standard read-outs for asking "what feature of the stimulus does
this neuron encode, and how reliably?"
"""

from __future__ import annotations

import numpy as np


def firing_rate(spike_times: np.ndarray, duration_s: float,
               bin_ms: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Binned firing rate (Hz). Returns (bin_centers_s, rate_hz)."""
    bin_s = bin_ms / 1000.0
    edges = np.arange(0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times, bins=edges)
    rate = counts / bin_s
    centers = edges[:-1] + bin_s / 2
    return centers, rate


def spike_phases(spike_times: np.ndarray, stim_freq_hz: float) -> np.ndarray:
    """Phase (radians, 0-2pi) of each spike within the stimulus cycle."""
    period = 1.0 / stim_freq_hz
    return (spike_times % period) / period * 2 * np.pi


def phase_histogram(spike_times: np.ndarray, stim_freq_hz: float,
                   n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Cycle-folded phase histogram. Returns (bin_centers_rad, counts)."""
    phases = spike_phases(spike_times, stim_freq_hz)
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    counts, _ = np.histogram(phases, bins=edges)
    centers = edges[:-1] + np.diff(edges) / 2
    return centers, counts


def vector_strength(spike_times: np.ndarray, stim_freq_hz: float) -> dict:
    """Vector strength (r), preferred phase, and Rayleigh test p-value.

    r ranges 0 (no phase locking) to 1 (perfect locking). The Rayleigh test
    asks whether the phase distribution differs from uniform.
    """
    phases = spike_phases(spike_times, stim_freq_hz)
    n = phases.size
    if n == 0:
        return {"r": 0.0, "preferred_phase": np.nan, "p_value": 1.0, "n": 0}
    c = np.mean(np.cos(phases))
    s = np.mean(np.sin(phases))
    r = np.hypot(c, s)
    preferred = np.arctan2(s, c) % (2 * np.pi)
    # Rayleigh test
    z = n * r ** 2
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n))
    p = float(np.clip(p, 0.0, 1.0))
    return {"r": r, "preferred_phase": preferred, "p_value": p, "n": n}


def isi_stats(spike_times: np.ndarray) -> dict:
    """Inter-spike interval summary (ms) and coefficient of variation."""
    if spike_times.size < 2:
        return {"isi_ms": np.array([]), "mean_ms": np.nan, "cv": np.nan}
    isi = np.diff(np.sort(spike_times)) * 1000.0
    cv = np.std(isi) / np.mean(isi) if np.mean(isi) else np.nan
    return {"isi_ms": isi, "mean_ms": float(np.mean(isi)), "cv": float(cv)}


def mean_rate(spike_times: np.ndarray, duration_s: float) -> float:
    return spike_times.size / duration_s if duration_s else 0.0
