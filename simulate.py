"""
simulate.py
-----------
Generate a synthetic extracellular recording from a single mechanosensory
neuron whose firing is driven by a sinusoidal "wing-beat" stimulus.

This stands in for a real electrophysiology trace so the analysis pipeline can
be demonstrated end-to-end without proprietary data. Swap `simulate_recording`
for a loader (e.g. Neo / NWB / .abf) to run the same pipeline on real recordings.
"""

from __future__ import annotations

import numpy as np


def simulate_stimulus(
    duration_s: float,
    fs: float,
    freq_hz: float = 25.0,
    amplitude: float = 1.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, stimulus) for a sinusoidal mechanical drive.

    Models the oscillating mechanical load a wing-hinge sensor experiences
    during flight at `freq_hz` wing-beats per second.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    stimulus = amplitude * np.sin(2 * np.pi * freq_hz * t)
    # small mechanical noise on the stimulus itself
    stimulus = stimulus + 0.02 * rng.standard_normal(t.size)
    return t, stimulus


def _spike_waveform(fs: float, dur_ms: float = 2.0) -> np.ndarray:
    """A simple biphasic extracellular spike template."""
    n = int(fs * dur_ms / 1000.0)
    x = np.linspace(-3, 3, n)
    # negative-going main deflection with a positive rebound
    w = -np.exp(-(x ** 2)) + 0.4 * np.exp(-((x - 1.2) ** 2) / 0.5)
    return w / np.max(np.abs(w))


def simulate_recording(
    duration_s: float = 10.0,
    fs: float = 20000.0,
    stim_freq_hz: float = 25.0,
    base_rate_hz: float = 5.0,
    gain: float = 60.0,
    noise_uv: float = 12.0,
    spike_amp_uv: float = 90.0,
    seed: int | None = 0,
) -> dict:
    """Simulate an extracellular voltage trace driven by a mechanical stimulus.

    The neuron is modelled as an inhomogeneous Poisson process whose
    instantaneous rate is a rectified, phase-shifted function of the stimulus,
    so spikes are locked to a preferred phase of the wing-beat cycle.

    Returns a dict with keys: fs, t, stimulus, voltage, spike_times, rate.
    """
    rng = np.random.default_rng(seed)
    t, stimulus = simulate_stimulus(duration_s, fs, stim_freq_hz, seed=seed)
    dt = 1.0 / fs

    # Rate is driven by the positive half-wave of the stimulus (rectification),
    # phase-shifted to give the neuron a preferred response phase.
    drive = np.sin(2 * np.pi * stim_freq_hz * t - np.pi / 4)
    rate = base_rate_hz + gain * np.clip(drive, 0, None)  # Hz, >= base_rate

    # Inhomogeneous Poisson spike train via thinning on the time grid.
    spike_prob = rate * dt
    spikes = rng.random(t.size) < spike_prob
    spike_idx = np.flatnonzero(spikes)

    # Build the voltage trace: noise + spike template at each spike index.
    voltage = noise_uv * rng.standard_normal(t.size)
    template = _spike_waveform(fs) * spike_amp_uv
    half = template.size // 2
    for idx in spike_idx:
        lo = idx - half
        hi = lo + template.size
        if lo < 0 or hi > voltage.size:
            continue
        voltage[lo:hi] += template

    return {
        "fs": fs,
        "t": t,
        "stimulus": stimulus,
        "voltage": voltage,
        "spike_times": spike_idx / fs,
        "rate": rate,
        "stim_freq_hz": stim_freq_hz,
    }


if __name__ == "__main__":
    rec = simulate_recording()
    print(f"Simulated {rec['t'][-1]:.1f} s at {rec['fs']:.0f} Hz")
    print(f"True spikes: {rec['spike_times'].size} "
          f"({rec['spike_times'].size / rec['t'][-1]:.1f} Hz mean rate)")
