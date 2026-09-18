"""Checks the ctypes binding against the C unit tests in
components/audio/test/test_frontend.c

    python -m frontend.selftest
"""

import numpy as np

from .fe import BLOCK_HZ, BLOCK_SAMPLES, SAMPLE_RATE, SPEC_VERSION, Frontend


def sine(freq, blocks):
    t = np.arange(blocks * BLOCK_SAMPLES) / SAMPLE_RATE
    return (30000.0 * np.sin(2 * np.pi * freq * t)).astype("<i2")


def run_sine(notch_hz, freq, blocks):
    return Frontend(notch_hz).run(sine(freq, blocks))[-1]


def main():
    on = run_sine(240, 240.0, 40)
    off = run_sine(480, 240.0, 40)
    assert on["rms"] < 0.02 * off["rms"], (on["rms"], off["rms"])

    p = run_sine(480, 1000.0, 40)
    s = run_sine(480, 6000.0, 40)
    assert s["rms"] < 0.2 * p["rms"], (p["rms"], s["rms"])

    a = run_sine(480, 1000.0, 12)
    b = run_sine(480, 1000.0, 12)
    assert a == b, (a, b)

    fe = Frontend(480)
    whole = fe.run(sine(1000.0, 8))
    fe2 = Frontend(480)
    split = np.concatenate([fe2.run(sine(1000.0, 8)[: 4 * BLOCK_SAMPLES]),
                            fe2.run(sine(1000.0, 8)[4 * BLOCK_SAMPLES:])])
    assert np.array_equal(whole, split), "state must carry across run() calls"

    print(f"frontend spec v{SPEC_VERSION}  {SAMPLE_RATE} Hz  "
          f"{BLOCK_SAMPLES} samples/block  {BLOCK_HZ:.3f} blocks/s")
    print(f"comb null {on['rms']:.2e} vs {off['rms']:.2e} | "
          f"hi-cut {s['rms']:.2e} vs {p['rms']:.2e}")
    print("OK")


if __name__ == "__main__":
    main()
