"""
run_pipeline.py
---------------
End-to-end demo: simulate a recording, detect spikes, quantify how the
stimulus is encoded, and save publication-style figures to ./figures.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --duration 20 --stim-freq 30
"""

from __future__ import annotations

import argparse
import os
import sys

# make src/ importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

import analysis  # noqa: E402
import plotting  # noqa: E402
import simulate  # noqa: E402
import spike_detection as sd  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Sensory-encoding ephys pipeline")
    p.add_argument("--duration", type=float, default=10.0, help="seconds")
    p.add_argument("--fs", type=float, default=20000.0, help="sample rate Hz")
    p.add_argument("--stim-freq", type=float, default=25.0, help="stimulus Hz")
    p.add_argument("--threshold-k", type=float, default=4.0,
                   help="detection threshold in noise sigmas")
    p.add_argument("--outdir", type=str, default="figures")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("1. Simulating recording...")
    rec = simulate.simulate_recording(
        duration_s=args.duration, fs=args.fs, stim_freq_hz=args.stim_freq)
    print(f"   {rec['t'][-1]:.1f}s @ {rec['fs']:.0f}Hz; "
          f"{rec['spike_times'].size} ground-truth spikes")

    print("2. Detecting spikes (bandpass -> MAD threshold -> peak find)...")
    det = sd.detect_spikes(rec["voltage"], rec["fs"],
                           threshold_k=args.threshold_k)
    print(f"   detected {det['spike_times'].size} spikes "
          f"(threshold {det['threshold']:.0f} uV)")
    report = sd.detection_report(rec["spike_times"], det["spike_times"],
                                 fs=rec["fs"])
    print(f"   precision {report['precision']:.2f}, "
          f"recall {report['recall']:.2f}, F1 {report['f1']:.2f}")

    print("3. Quantifying stimulus encoding...")
    centers, rate = analysis.firing_rate(det["spike_times"], args.duration)
    ph_centers, ph_counts = analysis.phase_histogram(
        det["spike_times"], args.stim_freq)
    vs = analysis.vector_strength(det["spike_times"], args.stim_freq)
    isi = analysis.isi_stats(det["spike_times"])
    mr = analysis.mean_rate(det["spike_times"], args.duration)
    print(f"   mean rate {mr:.1f} Hz")
    print(f"   vector strength r={vs['r']:.2f}, "
          f"preferred phase {np.degrees(vs['preferred_phase']):.0f} deg, "
          f"Rayleigh p={vs['p_value']:.1e}")
    print(f"   ISI mean {isi['mean_ms']:.1f} ms, CV {isi['cv']:.2f}")

    print("4. Saving figures...")
    plotting.plot_trace(rec["t"], rec["voltage"], det["filtered"],
                        det["threshold"], det["spike_times"], rec["fs"],
                        os.path.join(args.outdir, "01_trace_detection.png"))
    plotting.plot_psth(rec["t"], rec["stimulus"], centers, rate,
                      os.path.join(args.outdir, "02_stimulus_psth.png"))
    plotting.plot_phase_tuning(ph_centers, ph_counts, vs,
                              os.path.join(args.outdir, "03_phase_tuning.png"))
    plotting.plot_isi(isi["isi_ms"], isi["cv"],
                     os.path.join(args.outdir, "04_isi_distribution.png"))
    print(f"   wrote 4 figures to ./{args.outdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
