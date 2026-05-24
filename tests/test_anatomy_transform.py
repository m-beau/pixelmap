"""Tests for the probe-local → atlas coordinate transform.

These tests use small synthetic electrode layouts so the expected atlas
positions can be reasoned about by hand and asserted exactly (with float
tolerance). The full Allen CCF lookup path is exercised separately; this
file is purely about geometry.
"""

import numpy as np
import pytest

from pixelmap.anatomy.transform import probe_to_atlas


TOL = 1e-9


class TestVerticalProbe:
    """yaw = pitch = 0 → probe is vertical with shank-line along +ML."""

    def test_tip_electrode_lands_on_tip(self):
        # Lowest electrode at probe (0, 0) must map exactly to tip_atlas.
        electrode_xy = np.array([[0.0, 0.0]])
        tip = (5000.0, 2500.0, 3000.0)  # (AP, ML, DV)

        out = probe_to_atlas(electrode_xy, tip)
        np.testing.assert_allclose(out[0], tip, atol=TOL)

    def test_going_up_the_shank_decreases_DV(self):
        # +yp moves dorsal in atlas, i.e. DV decreases.
        electrode_xy = np.array([[0.0, 100.0]])  # 100 µm up the shank
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip)
        np.testing.assert_allclose(out[0], (5000.0, 2500.0, 2900.0), atol=TOL)

    def test_shank_line_runs_along_ML(self):
        # +xp at yaw=0 means +ML (lateral, to the right of the midline).
        electrode_xy = np.array([[250.0, 0.0]])  # 250 µm to the right
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip)
        np.testing.assert_allclose(out[0], (5000.0, 2750.0, 3000.0), atol=TOL)

    def test_multi_shank_layout_at_default_pose(self):
        # All shank tips lie on a horizontal line at the same DV.
        electrode_xy = np.array([
            [0.0, 0.0],
            [250.0, 0.0],
            [500.0, 0.0],
            [750.0, 0.0],
        ])
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip)
        np.testing.assert_allclose(out[:, 0], 5000.0, atol=TOL)              # AP fixed
        np.testing.assert_allclose(out[:, 1], [2500, 2750, 3000, 3250], atol=TOL)  # ML steps
        np.testing.assert_allclose(out[:, 2], 3000.0, atol=TOL)              # DV fixed


class TestYaw:
    """Yaw rotates the probe about the DV axis. Shank-line direction changes."""

    def test_yaw_90_aligns_shanks_with_AP(self):
        electrode_xy = np.array([[250.0, 0.0]])
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip, pitch_deg=0, yaw_deg=90)
        # +xp now points along +AP.
        np.testing.assert_allclose(out[0], (5250.0, 2500.0, 3000.0), atol=1e-6)

    def test_yaw_does_not_affect_vertical_axis(self):
        # Going up the shank should still be purely dorsal regardless of yaw.
        electrode_xy = np.array([[0.0, 100.0]])
        tip = (5000.0, 2500.0, 3000.0)

        for yaw in (-45, 0, 30, 90, 180):
            out = probe_to_atlas(electrode_xy, tip, yaw_deg=yaw)
            np.testing.assert_allclose(out[0, 2], 2900.0, atol=1e-6,
                                       err_msg=f"yaw={yaw}")


class TestPitch:
    """Pitch tilts the probe in the AP-DV plane (around probe-x after yaw)."""

    def test_pitch_90_lays_probe_horizontal_along_AP(self):
        # With pitch=90 and yaw=0, the shank lies horizontally pointing forward.
        electrode_xy = np.array([[0.0, 100.0]])
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip, pitch_deg=90, yaw_deg=0)
        np.testing.assert_allclose(out[0], (5100.0, 2500.0, 3000.0), atol=1e-6)

    def test_pitch_does_not_change_shank_line_direction(self):
        # +xp displacement should still be purely along the (yawed) shank-line
        # direction even with pitch applied; pitch rotates about probe-x.
        electrode_xy = np.array([[250.0, 0.0]])
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip, pitch_deg=45, yaw_deg=0)
        np.testing.assert_allclose(out[0], (5000.0, 2750.0, 3000.0), atol=1e-6)

    def test_positive_pitch_tips_top_forward(self):
        # Going up the shank (+yp) with positive pitch should move toward +AP.
        electrode_xy = np.array([[0.0, 100.0]])
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip, pitch_deg=30, yaw_deg=0)
        assert out[0, 0] > 5000.0, "expected anterior shift with +pitch"
        assert out[0, 2] < 3000.0, "expected dorsal lift with +pitch"


class TestComposedPose:
    """Yaw + pitch combined: verify the composition order matches the docstring."""

    def test_yaw90_pitch90_lays_probe_horizontal_along_ML(self):
        # Yaw=90 rotates shank-line to +AP. Then pitch=90 lays the probe
        # horizontal, with the top of the probe pointing toward where +AP
        # would have rotated *under* pitch — i.e., still in the horizontal
        # plane perpendicular to the shank line. The probe-y axis ends up
        # pointing along +ML after yaw=90 + pitch=90.
        electrode_xy = np.array([[0.0, 100.0]])
        tip = (5000.0, 2500.0, 3000.0)

        out = probe_to_atlas(electrode_xy, tip, pitch_deg=90, yaw_deg=90)
        # After yaw=90: horizontal_forward = (cos 90, -sin 90, 0) = (0, -1, 0) → -ML
        # With pitch=90, probe-y = (0, -1, 0): going up the shank moves toward -ML.
        np.testing.assert_allclose(out[0], (5000.0, 2400.0, 3000.0), atol=1e-6)


class TestInputValidation:
    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError):
            probe_to_atlas(np.array([1.0, 2.0, 3.0]), tip_atlas=(0, 0, 0))

    def test_list_input_accepted(self):
        out = probe_to_atlas([[0.0, 0.0]], tip_atlas=(1.0, 2.0, 3.0))
        np.testing.assert_allclose(out[0], (1.0, 2.0, 3.0), atol=TOL)
