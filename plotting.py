"""
plotting.py
-----------
Publication-style figures for the analysis pipeline. Every function takes an
output path and saves a tight-bbox PNG at 200 dpi.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def plot_trace(t, voltage, filtered, threshold, spike_times, fs, out_path,
               window_s=(0.0, 0.5)):
    """Raw + filtered trace with detected spikes over a short window."""
    lo, hi = window_s
    m = (t >= lo) & (t < hi)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(t[m], voltage[m], lw=0.6, color="#444")
    axes[0].set_ylabel("Raw (uV)")
    axes[0].set_title("Extracellular recording")
    axes[1].plot(t[m], filtered[m], lw=0.6, color="#1f77b4")
    axes[1].axhline(-threshold, ls="--", lw=0.8, color="#d62728",
                    label=f"threshold ({-threshold:.0f} uV)")
    sm = (spike_times >= lo) & (spike_times < hi)
    axes[1].plot(spike_times[sm], np.full(sm.sum(), -threshold * 1.3),
                 "v", ms=5, color="#d62728", label="detected spike")
    axes[1].set_ylabel("Filtered (uV)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_psth(t, stimulus, centers, rate, out_path):
    """Stimulus aligned above the firing-rate PSTH."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(t, stimulus, lw=0.7, color="#2ca02c")
    axes[0].set_ylabel("Stimulus (a.u.)")
    axes[0].set_title("Stimulus and firing-rate response")
    axes[1].bar(centers, rate, width=(centers[1] - centers[0]) * 0.9,
                color="#1f77b4", align="center")
    axes[1].set_ylabel("Firing rate (Hz)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_tuning(centers_rad, counts, vs, out_path):
    """Polar phase-tuning plot with the vector-strength resultant."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    width = (2 * np.pi) / len(centers_rad)
    ax.bar(centers_rad, counts, width=width, bottom=0.0,
           color="#1f77b4", alpha=0.7, edgecolor="white")
    # resultant vector scaled to the max bar height
    if vs["n"] > 0 and np.isfinite(vs["preferred_phase"]):
        ax.annotate("", xy=(vs["preferred_phase"], vs["r"] * counts.max()),
                    xytext=(0, 0),
                    arrowprops=dict(color="#d62728",
                                    arrowstyle="-|>", lw=2))
    ax.set_theta_zero_location("E")
    ax.set_title(f"Phase tuning\nvector strength r = {vs['r']:.2f}, "
                 f"p = {vs['p_value']:.1e}", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_isi(isi_ms, cv, out_path):
    """Inter-spike-interval distribution."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if isi_ms.size:
        ax.hist(isi_ms, bins=40, color="#9467bd", alpha=0.8)
    ax.set_xlabel("Inter-spike interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"ISI distribution (CV = {cv:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
