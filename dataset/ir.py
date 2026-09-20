"""Impulse response measurement: sweep generation and deconvolution.

Farina exponential sine sweep. The ESS is used rather than white noise or an MLS
because its deconvolution pushes harmonic distortion products *before* the linear
impulse in time, so a speaker driven hard at the bottom of its range does not
smear the measurement — it just leaves artefacts that get windowed off.

What this measures is the whole chain: speaker -> room -> mic. That is the right
transfer function for this project. The device hears music played through
speakers in a room, so convolving clean corpus samples with it reproduces what
the device actually hears, rather than what a studio monitor path would.

    python -m dataset.ir sweep   --out data/ir/sweep.wav
    python -m dataset.ir extract --rec data/ir/recording.wav --out data/ir/mic.wav
    python -m dataset.ir adopt   --ir rew_export.wav --out data/ir/mic.wav
    python -m dataset.ir selftest

`adopt` takes an IR produced elsewhere (REW's "Import Sweep Recordings" path,
say) and applies the same windowing and normalisation as `extract`, so either
route ends at the same artefact. An externally produced IR usually has its direct
sound at an arbitrary offset; that is fine, the peak is located here.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 48000          # match the device exactly; no resampling in the measurement path
F0, F1 = 20.0, 20000.0
DURATION = 15.0     # log sweep spends most of its time low, where SNR is worst
LEAD_S = 1.0        # silence before, so the recording start is never clipped
TAIL_S = 3.0        # silence after, long enough for the room to decay
# 10 ms: long enough that the sweep does not start on a step and ring the woofer,
# short enough not to eat the bottom octave. At 50 ms the fade spans a whole cycle
# at 20 Hz and costs 5x the in-band error (measured 2026-09-20).
FADE_S = 0.01


@dataclass
class SweepSpec:
    sr: int = SR
    f0: float = F0
    f1: float = F1
    duration: float = DURATION
    lead_s: float = LEAD_S
    tail_s: float = TAIL_S
    fade_s: float = FADE_S


def sweep(spec: SweepSpec = SweepSpec()) -> tuple[np.ndarray, np.ndarray]:
    """-> (sweep, inverse filter). Convolving the two gives a unit impulse."""
    n = int(spec.duration * spec.sr)
    t = np.arange(n) / spec.sr
    k = np.log(spec.f1 / spec.f0)
    x = np.sin(2 * np.pi * spec.f0 * spec.duration / k * (np.exp(t * k / spec.duration) - 1.0))

    f = int(spec.fade_s * spec.sr)
    if f > 0:
        w = 0.5 * (1 - np.cos(np.pi * np.arange(f) / f))
        x[:f] *= w
        x[-f:] *= w[::-1]

    # Farina inverse: time-reverse and apply -6 dB/octave, which flattens the
    # sweep's own 1/f energy distribution.
    inv = x[::-1] * np.exp(-t * k / spec.duration)
    inv /= np.abs(np.fft.rfft(np.convolve(x, inv, mode="full"))).max() or 1.0
    return x.astype(np.float64), inv.astype(np.float64)


def stimulus(spec: SweepSpec = SweepSpec()) -> np.ndarray:
    """The file to play: silence, sweep, silence."""
    x, _ = sweep(spec)
    return np.concatenate(
        [np.zeros(int(spec.lead_s * spec.sr)), x, np.zeros(int(spec.tail_s * spec.sr))]
    )


def deconvolve(rec: np.ndarray, spec: SweepSpec = SweepSpec()) -> np.ndarray:
    """Recording -> full deconvolved response, linear IR preceded by distortion."""
    _, inv = sweep(spec)
    n = len(rec) + len(inv) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(rec, nfft) * np.fft.rfft(inv, nfft), nfft)[:n]
    return y


def extract(
    rec: np.ndarray,
    spec: SweepSpec = SweepSpec(),
    pre_ms: float = 5.0,
    length_ms: float = 500.0,
) -> tuple[np.ndarray, dict]:
    """Window the linear IR out of a deconvolved response.

    Harmonic distortion lands *before* the linear impulse, so the window starts
    only a few ms early. Everything before that is discarded on purpose.
    """
    y = deconvolve(rec, spec)
    ir, info = window_ir(y, spec.sr, pre_ms, length_ms)
    pre = int(pre_ms * spec.sr / 1000)
    start = max(0, info["peak_sample"] - pre)

    noise = float(np.sqrt(np.mean(y[: max(1, start - pre)] ** 2))) if start > 2 * pre else 0.0
    sig = float(np.sqrt(np.mean(ir**2)))
    return ir, {**info, "snr_db": float(20 * np.log10(sig / noise)) if noise > 0 else None}


def window_ir(
    ir: np.ndarray, sr: int = SR, pre_ms: float = 5.0, length_ms: float = 500.0
) -> tuple[np.ndarray, dict]:
    """Locate the direct sound, window, normalise. Shared by extract and adopt."""
    peak = int(np.argmax(np.abs(ir)))
    pre = int(pre_ms * sr / 1000)
    n = int(length_ms * sr / 1000)
    start = max(0, peak - pre)
    out = ir[start : start + n].copy()
    f = int(0.005 * sr)
    if len(out) > f:
        out[-f:] *= np.linspace(1.0, 0.0, f)
    peak_val = float(np.abs(out).max())
    if peak_val > 0:
        out /= peak_val
    return out, {"peak_sample": peak, "direct_at_ms": pre_ms, "length_ms": length_ms}


def response_db(ir: np.ndarray, sr: int = SR, bands: int = 24) -> list[tuple[float, float]]:
    """Coarse magnitude response, for eyeballing that a measurement is sane."""
    H = np.abs(np.fft.rfft(ir))
    freqs = np.fft.rfftfreq(len(ir), 1 / sr)
    edges = np.geomspace(20, min(20000, sr / 2 * 0.95), bands + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            out.append((float(np.sqrt(lo * hi)), float(20 * np.log10(H[m].mean() + 1e-12))))
    ref = np.median([d for _, d in out])
    return [(f, d - ref) for f, d in out]


def _selftest() -> int:
    """Recover a known IR through the full path, with noise and a level offset.

    The recovered IR is band-limited to the swept range by construction, so it is
    compared against the truth's magnitude response *inside* that band, not
    sample-by-sample against a full-band delta.
    """
    spec = SweepSpec(duration=8.0)
    rng = np.random.default_rng(0)
    truth = np.zeros(int(0.25 * spec.sr))
    truth[0] = 1.0
    for d, g in ((0.007, -0.45), (0.013, 0.3), (0.031, -0.2), (0.062, 0.12)):
        truth[int(d * spec.sr)] = g
    decay = np.exp(-np.arange(len(truth)) / (0.04 * spec.sr))
    truth += rng.normal(0, 0.02, len(truth)) * decay  # diffuse tail

    played = stimulus(spec)
    rec = np.convolve(played, truth)[: len(played) + len(truth)] * 0.3
    rec += rng.normal(0, 1e-4, len(rec))  # mic noise floor

    ir, info = extract(rec, spec, length_ms=250.0)
    got = dict(response_db(ir, spec.sr, bands=48))
    want = dict(response_db(truth / np.abs(truth).max(), spec.sr, bands=48))
    err = lambda lo, hi: [abs(got[f] - want[f]) for f in got if lo <= f <= hi]

    # 60 Hz - 6 kHz is what the device actually uses: its lowest sub-band starts
    # at 40 Hz and the front end hi-cuts at 4 kHz. Accuracy degrades within about
    # a third of an octave of f0, which is why the sweep starts well below 40 Hz.
    worst = max(err(60, 6000))
    median = float(np.median(err(30, 10000)))

    lag = int(np.argmax(np.correlate(ir[int(0.005 * spec.sr):], truth, mode="full")))
    lag -= len(truth) - 1
    print(f"magnitude error 60 Hz-6 kHz: {worst:.3f} dB max")
    print(f"magnitude error 30 Hz-10 kHz: {median:.3f} dB median")
    print(f"alignment error: {lag} samples ({lag / spec.sr * 1000:+.3f} ms)")
    print(f"snr: {info['snr_db']:.1f} dB")
    ok = worst < 0.5 and median < 0.05 and lag == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("sweep", help="write the stimulus to play")
    g.add_argument("--out", type=Path, default=Path("data/ir/sweep.wav"))
    g.add_argument("--duration", type=float, default=DURATION)
    g.add_argument("--f0", type=float, default=F0)
    g.add_argument("--f1", type=float, default=F1)
    e = sub.add_parser("extract", help="deconvolve a recording into an IR")
    e.add_argument("--rec", type=Path, required=True)
    e.add_argument("--out", type=Path, default=Path("data/ir/mic.wav"))
    e.add_argument("--duration", type=float, default=DURATION)
    e.add_argument("--f0", type=float, default=F0)
    e.add_argument("--f1", type=float, default=F1)
    e.add_argument("--length-ms", type=float, default=500.0)
    e.add_argument("--rate", type=float, default=None,
                   help="true capture rate, if the device clock differs from the header")
    d = sub.add_parser("adopt", help="normalise an IR produced by REW or similar")
    d.add_argument("--ir", type=Path, required=True)
    d.add_argument("--out", type=Path, default=Path("data/ir/mic.wav"))
    d.add_argument("--length-ms", type=float, default=500.0)
    sub.add_parser("selftest")
    a = ap.parse_args()

    if a.cmd == "selftest":
        raise SystemExit(_selftest())

    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.cmd == "adopt":
        raw, sr = sf.read(a.ir, dtype="float64", always_2d=True)
        if sr != SR:
            raise SystemExit(f"IR is {sr} Hz, expected {SR}; re-export at {SR}")
        ir, info = window_ir(raw[:, 0], sr, length_ms=a.length_ms)
        sf.write(a.out, ir.astype(np.float32), sr, subtype="PCM_24")
        print(json.dumps(info, indent=2))
        print(f"\n{a.out}  {len(ir) / sr * 1000:.0f} ms")
        print("\nmagnitude response (dB, median-referenced):")
        for f, db in response_db(ir, sr):
            print(f"  {f:8.0f} Hz {db:+7.1f}  {'#' * max(0, int(round(db + 20)))}")
        return

    spec = SweepSpec(duration=a.duration, f0=a.f0, f1=a.f1)

    if a.cmd == "sweep":
        x = stimulus(spec) * 0.5  # -6 dBFS: headroom for the speaker, not the mic
        sf.write(a.out, x.astype(np.float32), spec.sr, subtype="PCM_24")
        (a.out.with_suffix(".json")).write_text(json.dumps(asdict(spec), indent=2))
        print(f"{a.out}  {len(x)/spec.sr:.1f}s  {spec.f0:.0f}-{spec.f1:.0f} Hz @ {spec.sr} Hz")
        print("play at 48 kHz, no resampling, no dither, no limiter")
        return

    rec, sr = sf.read(a.rec, dtype="float64", always_2d=True)
    x = rec[:, 0]
    x = x - x.mean()  # raw captures carry the mic's DC offset; the device's DC
                      # blocker has not run on these
    rate = a.rate or float(sr)
    if abs(rate - spec.sr) > 0.01:
        # The device clock is its own reference; the sweep was played at 48 kHz.
        # Left uncorrected, the mismatch stretches the sweep and smears the IR.
        drift_ms = (rate - spec.sr) / spec.sr * spec.duration * 1000
        print(f"capture rate {rate:.2f} Hz vs {spec.sr}: resampling "
              f"({drift_ms:+.2f} ms of drift across the sweep)")
        x = soxr.resample(x, rate, spec.sr, quality="VHQ")
    ir, info = extract(x, spec, length_ms=a.length_ms)
    sf.write(a.out, ir.astype(np.float32), spec.sr, subtype="PCM_24")
    print(json.dumps(info, indent=2))
    print(f"\n{a.out}  {len(ir)/spec.sr*1000:.0f} ms")
    print("\nmagnitude response (dB, median-referenced):")
    for f, d in response_db(ir, spec.sr):
        bar = "#" * max(0, int(round(d + 20)))
        print(f"  {f:8.0f} Hz {d:+7.1f}  {bar}")


if __name__ == "__main__":
    main()
