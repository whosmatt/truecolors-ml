"""Read Ableton's DRM-protected AIFC samples.

Ableton ships factory-pack samples as AIFC with compression type `able`. Nothing
is actually compressed: the SSND payload is ordinary PCM (its byte count equals
frames x channels x bytes-per-sample exactly) with a byte substitution applied
over it, and the `able` chunk carries the substitution table encrypted under a
fixed Blowfish key.

Format per https://github.com/adamchainz/abledecoder (MIT-era reference
implementation in C++; this is an independent Python reimplementation of the
same steps, verified byte-for-byte against that repo's example pair).

**Use within legal limits.**
"""

import struct

import numpy as np
from Crypto.Cipher import Blowfish

# Fixed across all files; lifted from the reference implementation.
_BF_KEY = bytes(
    (0x2B, 0xB1, 0x9D, 0x06, 0x98, 0xC3, 0xE1, 0xAB,
     0x20, 0xC6, 0xC1, 0x85, 0xFB, 0x7C, 0xD5, 0x17)
)
_BF_IV = bytes((0x28, 0x27, 0xC8, 0xE2, 0xC5, 0xEB, 0xA1, 0xB3))
_TABLE_OFF, _TABLE_LEN = 12, 256


class NotAbleFile(Exception):
    pass


def _chunks(data: bytes):
    if data[:4] != b"FORM" or data[8:12] != b"AIFC":
        raise NotAbleFile("not an AIFC file")
    i, end = 12, 8 + struct.unpack(">I", data[4:8])[0]
    while i + 8 <= min(end, len(data)):
        cid = data[i : i + 4]
        size = struct.unpack(">I", data[i + 4 : i + 8])[0]
        yield cid, data[i + 8 : i + 8 + size]
        i += 8 + size + (size & 1)


def _extended80(raw: bytes) -> int:
    """IEEE 754 80-bit extended, as AIFF stores the sample rate."""
    exp = struct.unpack(">H", raw[:2])[0] & 0x7FFF
    mant = struct.unpack(">Q", raw[2:10])[0]
    return int(mant * 2.0 ** (exp - 16383 - 63))


def _table(able: bytes) -> np.ndarray:
    length = struct.unpack(">I", able[12:16])[0]
    blob = able[16 : 16 + length]
    if length % 8 or len(blob) < length:
        raise NotAbleFile("malformed able chunk")
    # Trailing PKCS#7 padding is irrelevant: the table sits well before the end.
    plain = Blowfish.new(_BF_KEY, Blowfish.MODE_CBC, _BF_IV).decrypt(blob)
    if len(plain) < _TABLE_OFF + _TABLE_LEN:
        raise NotAbleFile("decrypted key too short")
    return np.frombuffer(plain[_TABLE_OFF : _TABLE_OFF + _TABLE_LEN], dtype=np.uint8)


def _deobfuscate(payload: bytes, table: np.ndarray) -> np.ndarray:
    """XOR each byte with table[b0^b1^b2^b3 of its index], leaving zeros and
    bytes already equal to their key byte untouched — the transform is its own
    inverse only under those two exemptions."""
    b = np.frombuffer(payload, dtype=np.uint8).copy()
    i = np.arange(b.size, dtype=np.uint32)
    idx = (i & 0xFF) ^ ((i >> 8) & 0xFF) ^ ((i >> 16) & 0xFF) ^ ((i >> 24) & 0xFF)
    k = table[idx.astype(np.uint8)]
    m = (b != 0) & (b != k)
    b[m] ^= k[m]
    return b


def info(path: str) -> tuple[int, int, int, int]:
    """-> (frames, channels, samplerate, bits) from COMM alone, no decryption."""
    with open(path, "rb") as fh:
        data = fh.read(4096)  # COMM precedes SSND in every file seen
    for cid, body in _chunks(data):
        if cid == b"COMM":
            channels, frames, bits = struct.unpack(">hIh", body[:8])
            return frames, channels, _extended80(body[8:18]), bits
    raise NotAbleFile("no COMM chunk")


def read(path: str) -> tuple[np.ndarray, int, str]:
    """-> (float32 samples (frames, channels), samplerate, subtype)."""
    data = open(path, "rb").read()
    comm = able = ssnd = None
    for cid, body in _chunks(data):
        if cid == b"COMM":
            comm = body
        elif cid == b"able":
            able = body
        elif cid == b"SSND":
            ssnd = body
    if comm is None or ssnd is None:
        raise NotAbleFile("missing COMM or SSND")
    channels, frames, bits = struct.unpack(">hIh", comm[:8])
    rate = _extended80(comm[8:18])
    compression = bytes(comm[18:22])
    payload = ssnd[8:]  # skip offset and blockSize

    if compression == b"able":
        if able is None:
            raise NotAbleFile("compression is able but no able chunk")
        raw = _deobfuscate(payload, _table(able)).tobytes()
    elif compression == b"NONE":
        raw = payload
    else:
        raise NotAbleFile(f"unsupported compression {compression!r}")

    if bits == 24:
        # Big-endian 24-bit widened into the HIGH three bytes of an int32, so the
        # sign survives; the implied <<8 is undone by the 2^31 divisor.
        a = np.frombuffer(raw[: frames * channels * 3], dtype=np.uint8).reshape(-1, 3)
        w = np.zeros((a.shape[0], 4), dtype=np.uint8)
        w[:, :3] = a
        x = w.view(">i4").reshape(-1).astype(np.float32) / 2147483648.0
    elif bits == 16:
        x = np.frombuffer(raw[: frames * channels * 2], dtype=">i2").astype(np.float32) / 32768.0
    elif bits == 32:
        x = np.frombuffer(raw[: frames * channels * 4], dtype=">i4").astype(np.float32) / 2147483648.0
    elif bits == 8:
        x = np.frombuffer(raw[: frames * channels], dtype=np.int8).astype(np.float32) / 128.0
    else:
        raise NotAbleFile(f"unsupported sample size {bits}")
    return x.reshape(-1, channels), rate, f"PCM_{bits}"
