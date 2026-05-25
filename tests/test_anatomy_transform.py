"""Tests for the probe-local → atlas coordinate transform.

These tests use small synthetic electrode layouts so the expected atlas
positions can be reasoned about by hand and asserted exactly (with float
tolerance). The full Allen CCF lookup path is exercised separately; this
file is purely about geometry.

Conventions (mirror the docstring of :mod:`pixelmap.anatomy.transform`):

* ``pitch_deg`` = AP tilt (around world ML).
* ``yaw_deg`` = ML tilt (around world AP).
* ``shank_orientation_deg`` = rotation of shank line around the probe's
  long axis; ``0°`` puts +xp along atlas +ML, ``90°`` along +AP.
"""

import numpy as np
import pytest

from pixelmap.anatomy.transform import probe_to_atlas


TOL = 1e-6


class TestVerticalProbe:
    """All angles 0 — probe straight down with shanks along +ML."""

    def test_tip_electrode_lands_on_tip(self):
        out = probe_to_atlas(np.array([[0.0, 0.0]]), tip_atlas=(5000, 2500, 3000))
        np.testing.assert_allclose(out[0], (5000, 2500, 3000), atol=TOL)

    def test_going_up_the_shank_decreases_DV(self):
        out = probe_to_atlas(np.array([[0.0, 100.0]]), tip_atlas=(5000, 2500, 3000))
        np.testing.assert_allclose(out[0], (5000, 2500, 2900), atol=TOL)

    def test_shank_line_runs_along_ML(self):
        out = probe_to_atlas(np.array([[250.0, 0.0]]), tip_atlas=(5000, 2500, 3000))
        np.testing.assert_allclose(out[0], (5000, 2750, 3000), atol=TOL)

    def test_multi_shank_tips_lie_on_horizontal_ML_line(self):
        electrode_xy = np.array([[0, 0], [250, 0], [500, 0], [750, 0]], dtype=float)
        out = probe_to_atlas(electrode_xy, tip_atlas=(5000, 2500, 3000))
        np.testing.assert_allclose(out[:, 0], 5000, atol=TOL)               # AP
        np.testing.assert_allclose(out[:, 1], [2500, 2750, 3000, 3250], atol=TOL)  # ML
        np.testing.assert_allclose(out[:, 2], 3000, atol=TOL)               # DV


class TestPitch:
    """AP tilt: rotation around world ML, mixes AP and DV."""

    def test_pitch_90_lays_probe_horizontal_along_AP(self):
        out = probe_to_atlas(np.array([[0.0, 100.0]]),
                             tip_atlas=(5000, 2500, 3000), pitch_deg=90)
        np.testing.assert_allclose(out[0], (5100, 2500, 3000), atol=TOL)

    def test_positive_pitch_tips_top_anterior(self):
        out = probe_to_atlas(np.array([[0.0, 100.0]]),
                             tip_atlas=(5000, 2500, 3000), pitch_deg=30)
        assert out[0, 0] > 5000, "expected +AP shift"
        assert out[0, 2] < 3000, "expected dorsal lift"

    def test_pitch_does_not_affect_ML(self):
        out = probe_to_atlas(np.array([[250.0, 0.0]]),
                             tip_atlas=(5000, 2500, 3000), pitch_deg=45)
        np.testing.assert_allclose(out[0, 1], 2750, atol=TOL)


class TestYaw:
    """ML tilt: rotation around world AP, mixes ML and DV."""

    def test_yaw_90_lays_probe_horizontal_along_ML(self):
        out = probe_to_atlas(np.array([[0.0, 100.0]]),
                             tip_atlas=(5000, 2500, 3000), yaw_deg=90)
        np.testing.assert_allclose(out[0], (5000, 2600, 3000), atol=TOL)

    def test_positive_yaw_tips_top_lateral(self):
        out = probe_to_atlas(np.array([[0.0, 100.0]]),
                             tip_atlas=(5000, 2500, 3000), yaw_deg=30)
        assert out[0, 1] > 2500, "expected +ML shift"
        assert out[0, 2] < 3000, "expected dorsal lift"

    def test_yaw_does_not_affect_AP(self):
        out = probe_to_atlas(np.array([[0.0, 100.0]]),
                             tip_atlas=(5000, 2500, 3000), yaw_deg=45)
        np.testing.assert_allclose(out[0, 0], 5000, atol=TOL)


class TestShankOrientation:
    """Rotation of the shank line around the probe long axis."""

    def test_default_0_keeps_shanks_along_ML(self):
        out = probe_to_atlas(np.array([[250.0, 0.0]]),
                             tip_atlas=(5000, 2500, 3000), shank_orientation_deg=0)
        np.testing.assert_allclose(out[0], (5000, 2750, 3000), atol=TOL)

    def test_90_puts_shanks_along_AP(self):
        out = probe_to_atlas(np.array([[250.0, 0.0]]),
                             tip_atlas=(5000, 2500, 3000), shank_orientation_deg=90)
        np.testing.assert_allclose(out[0], (5250, 2500, 3000), atol=TOL)

    def test_180_flips_shank_direction(self):
        out = probe_to_atlas(np.array([[250.0, 0.0]]),
                             tip_atlas=(5000, 2500, 3000), shank_orientation_deg=180)
        np.testing.assert_allclose(out[0], (5000, 2250, 3000), atol=TOL)

    def test_does_not_affect_along_shank_direction(self):
        # +yp displacement should still be purely -DV regardless of spin.
        for spin in (-30, 0, 30, 90, 180):
            out = probe_to_atlas(np.array([[0.0, 100.0]]),
                                 tip_atlas=(5000, 2500, 3000),
                                 shank_orientation_deg=spin)
            np.testing.assert_allclose(out[0], (5000, 2500, 2900), atol=TOL,
                                       err_msg=f"spin={spin}")


class TestComposedPose:
    """Pitch and yaw are independent of shank orientation."""

    def test_spin_then_pitch_preserves_pitch_geometry(self):
        # Spin the shank line to AP, then apply pitch. The shank line should
        # now lie in the AP-DV plane after pitch — but pitch only acts on
        # probe-y (around world ML), not on probe-x. So +xp at any pitch
        # should still be purely +AP.
        out = probe_to_atlas(np.array([[250.0, 0.0]]),
                             tip_atlas=(5000, 2500, 3000),
                             pitch_deg=45, shank_orientation_deg=90)
        # +xp = +AP after spin=90; pitch rotates probe-y but probe-x is the
        # rotation axis only when pitch were applied in probe frame. Here
        # pitch is in WORLD frame (around world ML), so it rotates +AP into
        # the AP-DV plane mixing with -DV.
        # +AP rotated around +ML by +pitch (using my R_pitch convention):
        # R_pitch maps (1, 0, 0) → (cos pitch, 0, sin pitch). At pitch=45,
        # that's (sqrt(2)/2, 0, sqrt(2)/2). Then +250 µm along that gives
        # AP +250*cos45 ≈ 176.78, DV +250*sin45 ≈ 176.78.
        s = 250 * np.sqrt(2) / 2
        np.testing.assert_allclose(out[0], (5000 + s, 2500, 3000 + s), atol=1e-4)

    def test_zero_orientation_matches_yp_only_under_yaw(self):
        # With shank_orientation=0, only yp contributes outside the shank
        # line — a yawed probe with xp=0 should produce the same result as
        # the simple yaw test (no spin in play).
        out_spin = probe_to_atlas(np.array([[0.0, 100.0]]),
                                  tip_atlas=(0, 0, 0),
                                  yaw_deg=30, shank_orientation_deg=0)
        out_no_spin = probe_to_atlas(np.array([[0.0, 100.0]]),
                                     tip_atlas=(0, 0, 0), yaw_deg=30)
        np.testing.assert_allclose(out_spin, out_no_spin, atol=TOL)


class TestInputValidation:
    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError):
            probe_to_atlas(np.array([1.0, 2.0, 3.0]), tip_atlas=(0, 0, 0))

    def test_list_input_accepted(self):
        out = probe_to_atlas([[0.0, 0.0]], tip_atlas=(1.0, 2.0, 3.0))
        np.testing.assert_allclose(out[0], (1.0, 2.0, 3.0), atol=TOL)
