"""Front-end features and block-aligned labels for a rendered take.

Features come from the firmware's own C front end via `frontend.fe`, never a
numpy reimplementation — the comb, hi-cut and per-band AGC are stateful and a
reimplementation would drift from the device invisibly.

The laser PWM frequency moves the comb notch at runtime, so every take is
featurised once per supported setting and the notch is carried as an input.
"""

import numpy as np

from frontend.fe import BLOCK_SAMPLES, DTYPE, SPEC_VERSION, Frontend, variant_name

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
NOTCH_HZ = (120, 240, 480)  # runtime-adjustable; dataset must cover all three


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


def featurise(
    pcm16: np.ndarray, notch_hz: int, comb: bool = True, hicut: bool = True
) -> np.ndarray:
    fe = Frontend(notch_hz=notch_hz, comb=comb, hicut=hicut)
    return flatten(fe.run(pcm16))


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


def spec(comb: bool = True, hicut: bool = True) -> dict:
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
    }
