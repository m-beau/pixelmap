"""Round-trip and robustness tests for the URL-share encoding."""

import pytest

from pixelmap.utils import url_share


def _sample_state():
    return dict(
        probe_type="2.0-4shanks",
        probe_subtype=24,
        reference_id="External",
        ap_gain=500.0,
        lf_gain=250.0,
        hp_filter=1,
        electrodes=[(s, e) for s in range(4) for e in range(96)],
    )


class TestRoundTrip:
    def test_full_state_round_trips(self):
        state = _sample_state()
        encoded = url_share.encode_state(**state)
        decoded = url_share.decode_state(encoded)

        assert decoded is not None
        assert decoded["probe_type"] == state["probe_type"]
        assert decoded["probe_subtype"] == state["probe_subtype"]
        assert decoded["reference_id"] == state["reference_id"]
        assert decoded["ap_gain"] == state["ap_gain"]
        assert decoded["lf_gain"] == state["lf_gain"]
        assert decoded["hp_filter"] == state["hp_filter"]
        assert decoded["electrodes"] == state["electrodes"]

    def test_empty_electrode_selection_round_trips(self):
        encoded = url_share.encode_state(
            probe_type="1.0",
            probe_subtype=0,
            reference_id="External",
            ap_gain=None,
            lf_gain=None,
            hp_filter=None,
            electrodes=[],
        )
        decoded = url_share.decode_state(encoded)

        assert decoded is not None
        assert decoded["electrodes"] == []
        assert decoded["ap_gain"] is None
        assert decoded["hp_filter"] is None

    def test_encoded_payload_is_url_safe(self):
        encoded = url_share.encode_state(**_sample_state())
        # urlsafe_b64encode uses [A-Za-z0-9_-]; we strip padding.
        assert all(c.isalnum() or c in "-_" for c in encoded)

    def test_encoded_payload_is_compact_for_full_selection(self):
        # A typical 384-electrode selection should comfortably fit in a URL.
        full = [(s, e) for s in range(4) for e in range(96)]
        encoded = url_share.encode_state(
            probe_type="2.0-4shanks",
            probe_subtype=24,
            reference_id="External",
            ap_gain=None,
            lf_gain=None,
            hp_filter=None,
            electrodes=full,
        )
        assert len(encoded) < 1500


class TestInvalidInput:
    @pytest.mark.parametrize("bad", ["", "   ", "not_base64!!!", "AAAA", "_____"])
    def test_garbage_strings_return_none(self, bad):
        assert url_share.decode_state(bad) is None

    def test_none_returns_none(self):
        assert url_share.decode_state(None) is None  # type: ignore[arg-type]

    def test_wrong_schema_version_returns_none(self):
        import base64
        import json
        import zlib

        bad_payload = {"v": 999, "pt": "1.0", "ps": 0, "ref": "External", "e": []}
        raw = json.dumps(bad_payload).encode("utf-8")
        encoded = base64.urlsafe_b64encode(zlib.compress(raw)).decode("ascii").rstrip("=")

        assert url_share.decode_state(encoded) is None

    def test_malformed_electrodes_list_returns_none(self):
        import base64
        import json
        import zlib

        bad_payload = {
            "v": 1,
            "pt": "1.0",
            "ps": 0,
            "ref": "External",
            "e": [[1, 2, 3]],  # wrong tuple length
        }
        raw = json.dumps(bad_payload).encode("utf-8")
        encoded = base64.urlsafe_b64encode(zlib.compress(raw)).decode("ascii").rstrip("=")

        assert url_share.decode_state(encoded) is None
