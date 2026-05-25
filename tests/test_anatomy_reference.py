"""Per-atlas reference params + the optional bregma→atlas transform.

Both are pure (no brainglobe needed): a name-keyed table and plain geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelmap.anatomy.atlas import reference_params
from pixelmap.anatomy.transform import bregma_to_atlas_um


def test_reference_params_matches_by_prefix():
    allen = reference_params("allen_mouse_10um")
    assert allen["bregma_um"] == (5200.0, 5705.0, 440.0)
    assert allen["ap_squish"] == 1.0  # no trusted estimate → default 1
    assert allen["ml_squish"] == 1.0
    assert allen["dv_squish"] == 0.885
    assert allen["tilt_deg"] == 15.0
    # every allen_mouse_* variant resolves to the same entry
    assert reference_params("allen_mouse_bluebrain_barrels_25um") is not None
    assert isinstance(allen["source"], str) and allen["source"]  # provenance text
    rat = reference_params("whs_sd_rat_39um")
    assert rat["bregma_um"] == (14469.0, 10374.0, 2808.0)
    assert isinstance(rat["source"], str) and rat["source"]
    assert rat["dv_squish"] == 1.0  # WHS already stereotaxically aligned
    # atlases without a published estimate
    assert reference_params("azba_zfish_4um") is None


def test_origin_maps_to_bregma():
    b = (5200.0, 5705.0, 440.0)
    assert bregma_to_atlas_um((0, 0, 0), bregma_um=b, dv_squish=0.885, tilt_deg=5.0) == b


def test_ml_offset_adds_to_atlas_ml():
    out = bregma_to_atlas_um((0, 1000, 0), bregma_um=(5200.0, 5705.0, 440.0),
                             dv_squish=1.0, tilt_deg=0.0)
    assert out == (5200.0, 6705.0, 440.0)  # +ML = left = larger atlas ML


def test_dv_squish_stretches_depth():
    # 885 µm real depth at squish 0.885 → 1000 µm of atlas DV (no tilt)
    out = bregma_to_atlas_um((0, 0, 885), bregma_um=(0.0, 0.0, 0.0),
                             dv_squish=0.885, tilt_deg=0.0)
    assert out[2] == pytest.approx(1000.0)
    assert out[0] == 0.0


def test_nose_up_tilt_lifts_anterior_point_dorsally():
    out = bregma_to_atlas_um((1000, 0, 0), bregma_um=(0.0, 0.0, 0.0),
                             dv_squish=1.0, tilt_deg=5.0)
    theta = np.deg2rad(5.0)
    assert out[0] == pytest.approx(-1000 * np.cos(theta))  # anterior → smaller atlas AP
    assert out[2] == pytest.approx(-1000 * np.sin(theta))  # lifted dorsally (−DV)
