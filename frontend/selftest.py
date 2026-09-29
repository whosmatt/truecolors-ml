"""Checks the ctypes binding against the C unit tests in
components/audio/test/test_frontend.c

    python -m frontend.selftest
"""

import numpy as np

from .fe import (BLOCK_HZ, BLOCK_SAMPLES, SAMPLE_RATE, SPEC_VERSION, VARIANT_COMB,
                 VARIANT_HICUT, Frontend)


def sine(freq, blocks):
    t = np.arange(blocks * BLOCK_SAMPLES) / SAMPLE_RATE
    return (30000.0 * np.sin(2 * np.pi * freq * t)).astype("<i2")


def run_sine(freq, blocks, hicut=True):
    return Frontend(hicut=hicut).run(sine(freq, blocks))[-1]


def check_mel():
    """melflux.c against the numpy prototype the corpora were built with."""
    from dataset import features
    from train.data import mel_view

    from .mel import MelFlux
    rng = np.random.default_rng(0)
    x = np.concatenate([sine(220.0, 40), (rng.standard_normal(40 * BLOCK_SAMPLES) * 8000).astype("<i2"),
                        sine(3000.0, 40)]).astype(np.float64)
    # A mic-like floor (-60 dBFS). Without it, bands ~110 dB under a full-scale tone sit
    # at float32 FFT rounding in the C code (float64 in numpy): 4.5e-3 apart, 1.2e-5 on
    # real audio (2026-09-29).
    x = np.clip(x + rng.standard_normal(len(x)) * 33.0, -32768, 32767).astype("<i2")
    c = MelFlux().run(x)
    n = len(c)
    ref = mel_view(np.concatenate([np.zeros((n, 12), np.float32), features.log_mel(x)], axis=1),
                   np.zeros(n, np.int32), "flux16")[:, 12:]
    err = float(np.abs(c - ref).max())
    assert err < 1e-3, err
    return err


def main():
    assert Frontend().variant == VARIANT_HICUT, Frontend().variant
    assert Frontend(hicut=False).variant == 0
    assert not Frontend().variant & VARIANT_COMB

    p = run_sine(1000.0, 40)
    s = run_sine(6000.0, 40)
    assert s["rms"] < 0.2 * p["rms"], (p["rms"], s["rms"])
    open_ = run_sine(6000.0, 40, hicut=False)
    assert open_["rms"] > 5 * s["rms"], (open_["rms"], s["rms"])

    a = run_sine(1000.0, 12)
    b = run_sine(1000.0, 12)
    assert a == b, (a, b)

    fe = Frontend()
    whole = fe.run(sine(1000.0, 8))
    fe2 = Frontend()
    split = np.concatenate([fe2.run(sine(1000.0, 8)[: 4 * BLOCK_SAMPLES]),
                            fe2.run(sine(1000.0, 8)[4 * BLOCK_SAMPLES:])])
    assert np.array_equal(whole, split), "state must carry across run() calls"

    print(f"frontend spec v{SPEC_VERSION}  {SAMPLE_RATE} Hz  "
          f"{BLOCK_SAMPLES} samples/block  {BLOCK_HZ:.3f} blocks/s")
    print(f"hi-cut 6 kHz {s['rms']:.2e} vs 1 kHz {p['rms']:.2e} | "
          f"open 6 kHz {open_['rms']:.2e}")
    print(f"melflux.c vs numpy flux16: max {check_mel():.2e}")
    print("OK")


if __name__ == "__main__":
    main()
