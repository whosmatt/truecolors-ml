"""Front-end features and block-aligned labels for a rendered take.

Features come from the firmware's own C front end via `frontend.fe`, never a
numpy reimplementation — the comb, hi-cut and per-band AGC are stateful and a
reimplementation would drift from the device invisibly.

The laser PWM setting changes the coil whine, so every take is rendered once
per setting with that setting's measured whine mixed in (`augment`), and the
setting is carried as `notch`. The front end itself no longer depends on it:
v2 removed the comb.
"""

import os

import numpy as np

from frontend.fe import BLOCK_SAMPLES, DTYPE, SAMPLE_RATE, SPEC_VERSION, Frontend, variant_name

# The flattening order below is part of the model contract: it goes into
# model_meta.json as feature_order and the firmware must fill its input tensor
# in exactly this order.
FEATURE_ORDER = (
    "level",
    "bands0", "bands1", "bands2",
    "rms",
    "flux0", "flux1", "flux2",
    "fund_rms",
    "mid_flux",
    "treble_flux",
    "spl_db",
)
N_FEATURES = len(FEATURE_ORDER)
NOTCH_HZ = (120, 240, 480)  # PWM settings with a measured whine capture


def flatten(blocks: np.ndarray) -> np.ndarray:
    """(n,) structured -> (n, 12) float32 in FEATURE_ORDER."""
    out = np.empty((blocks.shape[0], N_FEATURES), dtype=np.float32)
    out[:, 0] = blocks["level"]
    out[:, 1:4] = blocks["bands"]
    out[:, 4] = blocks["rms"]
    out[:, 5:8] = blocks["flux"]
    out[:, 8] = blocks["fund_rms"]
    out[:, 9] = blocks["mid_flux"]
    out[:, 10] = blocks["treble_flux"]
    out[:, 11] = blocks["spl_db"]
    return out


# TC_MEL=1 appends 40 raw log-mel columns to every build, so one render serves
# every band count, flux and normalisation variant, derived at load time
# (train.data). The flux16 view shipped: the device computes it in melflux.c
# (frontend/mel.py), and frontend.selftest checks the two agree.
MEL = os.environ.get("TC_MEL") == "1"
MEL_BANDS, MEL_LO_HZ, MEL_HI_HZ, MEL_WIN = 40, 150.0, 4000.0, 1024
# Below 150 Hz the FE's kick sub-bands resolve more than 47 Hz FFT bins can; the
# top is the FE's 4 kHz hi-cut.


def _mel_matrix() -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    edges = mel2hz(np.linspace(hz2mel(MEL_LO_HZ), hz2mel(MEL_HI_HZ), MEL_BANDS + 2))
    f = np.fft.rfftfreq(MEL_WIN, 1.0 / SAMPLE_RATE)
    W = np.zeros((len(f), MEL_BANDS), np.float32)
    for b in range(MEL_BANDS):
        lo, c, hi = edges[b : b + 3]
        W[:, b] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0.0, None)
    return W


def log_mel(pcm16: np.ndarray) -> np.ndarray:
    """-> (n_blocks, MEL_BANDS) log10 band power; block b's window ends with block b."""
    n = pcm16.size // BLOCK_SAMPLES
    x = np.concatenate([np.zeros(MEL_WIN - BLOCK_SAMPLES, np.float32),
                        pcm16[: n * BLOCK_SAMPLES].astype(np.float32) / 32768.0])
    frames = np.lib.stride_tricks.sliding_window_view(x, MEL_WIN)[::BLOCK_SAMPLES][:n]
    win, W = np.hanning(MEL_WIN).astype(np.float32), _mel_matrix()
    out = np.empty((n, MEL_BANDS), np.float32)
    for i in range(0, n, 4096):
        p = np.abs(np.fft.rfft(frames[i : i + 4096] * win, axis=1)) ** 2
        out[i : i + 4096] = np.log10(p.astype(np.float32) @ W + 1e-10)
    return out


def featurise(pcm16: np.ndarray, comb: bool = False, hicut: bool = True) -> np.ndarray:
    fe = Frontend(comb=comb, hicut=hicut)
    X = flatten(fe.run(pcm16))
    if MEL:
        X = np.concatenate([X, log_mel(pcm16)], axis=1)
    return X


def label_blocks(
    n_blocks: int, onsets: np.ndarray, classes: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """-> (hit (n_blocks, n_classes) float32, offset (n_blocks, n_classes) float32)

    `hit` marks the block an onset falls in. `offset` is where inside that block
    it landed, in [0, 1): a per-block label alone quantises onset time to
    +/-5.3 ms, so this is what a sub-block regression head would train against.
    Where two onsets of one class share a block, the earlier one wins.
    """
    hit = np.zeros((n_blocks, n_classes), dtype=np.float32)
    off = np.zeros((n_blocks, n_classes), dtype=np.float32)
    block = onsets // BLOCK_SAMPLES
    frac = (onsets % BLOCK_SAMPLES) / BLOCK_SAMPLES
    keep = block < n_blocks
    for b, f, c in zip(block[keep], frac[keep], classes[keep]):
        if hit[b, c] == 0.0 or f < off[b, c]:
            hit[b, c] = 1.0
            off[b, c] = f
    return hit, off


def spec(comb: bool = False, hicut: bool = True) -> dict:
    """Provenance for model_meta.json.

    `fe_variant` is as load-bearing as `fe_spec_version`: a model trained against
    one filter variant is invalid against another, and the mismatch is silent.
    """
    mask = Frontend(comb=comb, hicut=hicut).variant
    return {
        "fe_spec_version": SPEC_VERSION,
        "fe_variant": mask,
        "fe_variant_name": variant_name(mask),
        "feature_order": list(FEATURE_ORDER),
        "block_samples": BLOCK_SAMPLES,
        "notch_hz": list(NOTCH_HZ),
        **({"mel": {"bands": MEL_BANDS, "lo_hz": MEL_LO_HZ, "hi_hz": MEL_HI_HZ,
                    "window": MEL_WIN, "columns": "log10 power, after feature_order"}}
           if MEL else {}),
    }
