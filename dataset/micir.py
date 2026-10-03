"""Mic-only response of the device: device sweep / calibrated reference-mic sweep
taken side by side, so speaker and room cancel. Output is a minimum-phase IR of
the device mic alone, for convolving after a separate room IR.

    python -m dataset.micir data/ir/2026-10-03 --cal ".../39P380_cal_0degree.txt"

Expects, per take X: dev_X.raw (int16 raw capture from test/miccap) and
ref_X.wav (left = reference mic, played from data/ir/sweep.wav at 48 kHz on the
same interface clock). Takes named a*, b*, ... are positions of the reference
mic; dev_quiet.raw / ref_quiet.wav is a silent baseline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import ir as I

SR = 48000
NFFT = 1 << 16
GATE_MS = 100.0   # ratio is gate-independent above 150 Hz (5..500 ms within 0.5 dB, 2026-10-03)
SMOOTH_OCT = 12   # 1/12 octave: keeps the ~4 kHz port resonance's shape
F_LO = 25.0       # below this the sweep has no energy; response is held flat
# Above ~5.5 kHz the raw device capture is 2-6 dB over its PDM noise-shaping
# floor, and >14 kHz moved by ~40 dB between takes with nothing moved
# (2026-10-03). Far above the 4 kHz hi-cut, so held flat from here.
F_HI = 6000.0
F = np.fft.rfftfreq(NFFT, 1 / SR)


def _xcorr_lag(rec, chunk):
    n = len(rec) + len(chunk) - 1
    N = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(rec, N) * np.conj(np.fft.rfft(chunk, N)), N)[: len(rec)]
    i = int(np.argmax(np.abs(y)))
    y0, y1, y2 = np.abs(y[i - 1 : i + 2])
    return i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)


def device_rate(dev, spec):
    """Device clock against the playback clock, from where two sweep sections land.
    The device's own esp_timer estimate jitters by ~0.02% (miccap README)."""
    stim = I.stimulus(spec)
    t = lambda f: spec.lead_s + spec.duration * np.log(f / spec.f0) / np.log(spec.f1 / spec.f0)
    ta, tb = t(500), t(8000)
    la = _xcorr_lag(dev, stim[int(ta * SR) : int((ta + 1) * SR)]) - ta * SR
    lb = _xcorr_lag(dev, stim[int(tb * SR) : int((tb + 1) * SR)]) - tb * SR
    return SR * (1 + (lb - la) / ((tb - ta) * SR))


def load_dev(path):
    x = np.frombuffer(path.read_bytes(), "<i2").astype(np.float64) / 32768
    return x - x.mean()  # raw capture carries the mic's DC offset


def gated(y, p, ms=GATE_MS):
    pre, n = 48, int(ms * 48)
    s = y[p - pre : p + n].copy()
    s[:pre] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(pre) / pre)
    h = n // 4
    s[-h:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(h) / h)
    return np.fft.rfft(s, NFFT)


def smooth_db(db, frac=SMOOTH_OCT):
    p = 10 ** (db / 10)
    c = np.concatenate([[0], np.cumsum(p)])
    lo = np.searchsorted(F, F * 2 ** (-0.5 / frac))
    hi = np.maximum(np.searchsorted(F, F * 2 ** (0.5 / frac), "right"), lo + 1)
    return 10 * np.log10((c[hi] - c[lo]) / (hi - lo))


def take(d, name, spec, calf):
    dev = load_dev(d / f"dev_{name}.raw")
    ref = sf.read(d / f"ref_{name}.wav")[0][:, 0]
    rate = device_rate(dev, spec)
    yd = I.deconvolve(soxr.resample(dev, rate, SR, quality="VHQ"), spec)
    yr = I.deconvolve(ref, spec)
    pd, pr = int(np.argmax(np.abs(yd))), int(np.argmax(np.abs(yr)))
    D, R = gated(yd, pd), gated(yr, pr)
    out = {"rate": rate, "dev_db": 20 * np.log10(np.abs(D) + 1e-15),
           "ref_db": 20 * np.log10(np.abs(R) + 1e-15) - calf, "pd": pd, "pr": pr}
    out["mic_db"] = out["dev_db"] - out["ref_db"]
    return out


def psd_db(x, n=8192):
    """Welch PSD in dB on the NFFT grid."""
    w = np.hanning(n)
    segs = [x[i : i + n] * w for i in range(0, len(x) - n, n // 2)]
    p = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], 0)
    return 10 * np.log10(np.interp(F, np.fft.rfftfreq(n, 1 / SR), p) + 1e-30)


def snr_db(d, name, spec):
    """Band SNR of the raw recordings during the sweep against the silent
    baseline: before the deconvolution's own noise rejection, so a lower bound."""
    a, b = int((spec.lead_s + 0.5) * SR), int((spec.lead_s + spec.duration) * SR)
    out = []
    for sig, quiet in [(load_dev(d / f"dev_{name}.raw"), load_dev(d / "dev_quiet.raw")),
                       (sf.read(d / f"ref_{name}.wav")[0][:, 0], sf.read(d / "ref_quiet.wav")[0][:, 0])]:
        out.append(smooth_db(psd_db(sig[a:b]), 3) - smooth_db(psd_db(quiet), 3))
    return out


def min_phase(mag_db, n_out=4096):
    """Minimum-phase IR with the given magnitude (real cepstrum folding)."""
    logm = np.log(10 ** (mag_db / 20))
    c = np.fft.irfft(logm, NFFT)
    fold = np.zeros(NFFT)
    fold[0], fold[NFFT // 2] = c[0], c[NFFT // 2]
    fold[1 : NFFT // 2] = 2 * c[1 : NFFT // 2]
    h = np.fft.irfft(np.exp(np.fft.rfft(fold)), NFFT)[:n_out]
    f = n_out // 8
    h[-f:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(f) / f)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/ir/mic_only.wav"))
    ap.add_argument("--use", default=None,
                    help="reference positions to average (letters), default all")
    a = ap.parse_args()
    spec = I.SweepSpec(**json.loads(Path("data/ir/sweep.json").read_text()))
    cal = np.loadtxt(a.cal)
    calf = np.interp(F, cal[:, 0], cal[:, 1])
    names = sorted(p.stem[4:] for p in a.dir.glob("dev_*.raw") if p.stem != "dev_quiet")
    takes = {n: take(a.dir, n, spec, calf) for n in names}
    pos = sorted({n[0] for n in names})
    band = (F >= 900) & (F <= 1100)
    norm = lambda db: db - db[band].mean()

    per_pos = {p: norm(smooth_db(np.mean([takes[n]["mic_db"] for n in names if n[0] == p], 0)))
               for p in pos}
    use = list(a.use) if a.use else pos
    mic = norm(np.mean([per_pos[p] for p in use], 0))
    rep = {n: smooth_db(t["mic_db"]) for n, t in takes.items()}

    fq = [25, 30, 40, 50, 63, 80, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000,
          3500, 4000, 4500, 5000, 6000, 8000, 10000, 12000, 16000, 20000]
    idx = [int(np.argmin(np.abs(F - q))) for q in fq]
    print("device clock (Hz):", ", ".join(f"{n} {t['rate']:.2f}" for n, t in takes.items()))
    snr_d, snr_r = snr_db(a.dir, [n for n in names if n[0] in use][0], spec)
    spread = {p: np.ptp([norm(rep[n]) for n in names if n[0] == p], 0) for p in pos}
    hdr = "Hz      mic  " + "  ".join(f"pos {p}" for p in pos) + "  A-B   " + \
          "  ".join(f"rep {p}" for p in pos) + "  SNRdev SNRref"
    print(hdr)
    for q, i in zip(fq, idx):
        print(f"{q:6d} {mic[i]:+6.1f} " + " ".join(f"{per_pos[p][i]:+6.1f}" for p in pos)
              + f" {per_pos[pos[0]][i] - per_pos[pos[-1]][i]:+5.1f}  "
              + "  ".join(f"{spread[p][i]:5.2f}" for p in pos)
              + f"  {snr_d[i]:6.1f} {snr_r[i]:6.1f}")

    mag = mic.copy()
    lo, hi = np.searchsorted(F, F_LO), np.searchsorted(F, F_HI)
    mag[:lo], mag[hi:] = mag[lo], mag[hi - 1]
    h = min_phase(mag)
    h /= np.abs(h).max()
    sf.write(a.out, h.astype(np.float32), SR, subtype="FLOAT")
    np.savez(a.out.with_suffix(".npz"), f=F, mic_db=mic, **{f"pos_{p}": per_pos[p] for p in pos},
             snr_dev=snr_d, snr_ref=snr_r)
    meta = {"source": str(a.dir), "cal": a.cal.name, "takes": names, "positions_used": use, "gate_ms": GATE_MS,
            "smooth_oct": SMOOTH_OCT, "held_outside_hz": [F_LO, F_HI], "phase": "minimum",
            "normalised": "peak 1.0; response 0 dB at 1 kHz",
            "device_rates_hz": {n: round(t["rate"], 2) for n, t in takes.items()}}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", a.out, len(h), "taps")


if __name__ == "__main__":
    main()
