"""URL-shareable encoding of a channelmap selection state.

The encoded payload is a JSON object compressed with zlib and base64-encoded
(url-safe alphabet, no padding) so it can be passed as a single query-string
value. Decoding is strict: any malformed input returns ``None`` rather than
raising, so callers can fall back to the default state cleanly.
"""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any

QUERY_PARAM = "cfg"

# Bump this when the on-wire schema changes incompatibly.
_SCHEMA_VERSION = 1

# Hard cap on decoded payload size to prevent zip-bomb-style abuse on the
# server. A normal payload for 384 selected electrodes is a few KB.
_MAX_DECODED_BYTES = 256 * 1024


def encode_state(
    probe_type: str,
    probe_subtype: int,
    reference_id: str,
    ap_gain: float | None,
    lf_gain: float | None,
    hp_filter: int | None,
    electrodes: list[tuple[int, int]],
) -> str:
    """Encode a channelmap selection into a compact URL-safe string."""
    payload = {
        "v": _SCHEMA_VERSION,
        "pt": probe_type,
        "ps": int(probe_subtype),
        "ref": reference_id,
        "ag": ap_gain,
        "lg": lf_gain,
        "hp": hp_filter,
        "e": [[int(s), int(e)] for s, e in electrodes],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")


def decode_state(encoded: str) -> dict[str, Any] | None:
    """Decode a URL-shared state string. Returns ``None`` if invalid."""
    if not encoded or not isinstance(encoded, str):
        return None
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        compressed = base64.urlsafe_b64decode(padded.encode("ascii"))
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, _MAX_DECODED_BYTES)
        if decompressor.unconsumed_tail:
            return None  # payload exceeds the size cap
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, zlib.error, json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(payload, dict) or payload.get("v") != _SCHEMA_VERSION:
        return None
    if not isinstance(payload.get("pt"), str):
        return None

    electrodes_raw = payload.get("e", [])
    if not isinstance(electrodes_raw, list):
        return None
    electrodes: list[tuple[int, int]] = []
    for item in electrodes_raw:
        if not (isinstance(item, list) and len(item) == 2):
            return None
        try:
            electrodes.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            return None

    return {
        "probe_type": payload["pt"],
        "probe_subtype": payload.get("ps"),
        "reference_id": payload.get("ref"),
        "ap_gain": payload.get("ag"),
        "lf_gain": payload.get("lg"),
        "hp_filter": payload.get("hp"),
        "electrodes": electrodes,
    }
