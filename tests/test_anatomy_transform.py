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

from pixelmap.anatomy.transform import (
    atlas_to_bregma_um,
    bregma_to_atlas_um,
    insertion_depth_um,
    insertion_to_tip,
    probe_axis,
    probe_to_atlas,
    tip_to_insertion,
)


TOL = 1e-6

# Poses spanning both signs and the degenerate ±90° cases.
POSES = [(p, y) for p in (-45, -20, 0, 20, 45, 90) for y in (-30, 0, 30, 90)]
SPINS = (-30, 0, 90, 180)


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


class TestProbeAxis:
    """The along-shank unit vector, shared by probe_to_atlas and the
    insertion-point conversions."""

    def test_matches_probe_to_atlas_yp_displacement(self):
        # The load-bearing invariant: moving L µm up the shank must be the
        # same operation whether you go through probe_to_atlas or probe_axis.
        # This is what stops the insertion math drifting from the electrode
        # math, and it re-asserts spin-independence at the same time.
        tip = np.array([5000.0, 2500.0, 3000.0])
        L = 1234.5
        for pitch, yaw in POSES:
            for spin in SPINS:
                out = probe_to_atlas([[0.0, L]], tip, pitch, yaw, spin)[0]
                np.testing.assert_allclose(
                    out - tip, L * probe_axis(pitch, yaw), atol=1e-9,
                    err_msg=f"pitch={pitch} yaw={yaw} spin={spin}",
                )

    def test_closed_form(self):
        # Pins the derivation even though the implementation is matrix-based.
        for pitch, yaw in POSES:
            p, y = np.deg2rad(pitch), np.deg2rad(yaw)
            expected = (np.sin(p), np.sin(y) * np.cos(p), -np.cos(y) * np.cos(p))
            np.testing.assert_allclose(probe_axis(pitch, yaw), expected, atol=TOL)

    def test_is_unit_vector(self):
        for pitch, yaw in POSES:
            assert abs(np.linalg.norm(probe_axis(pitch, yaw)) - 1.0) < TOL

    def test_vertical_probe_points_dorsally(self):
        # Up the shank on an untilted probe is -DV (dorsal).
        np.testing.assert_allclose(probe_axis(0, 0), (0.0, 0.0, -1.0), atol=TOL)


class TestInsertionPoint:
    """Converting between the tip and the point where the shank enters."""

    TIP = (5000.0, 2500.0, 3000.0)

    def test_round_trip(self):
        for depth in (0.0, 10.0, 4000.0):
            for pitch, yaw in POSES:
                entry = tip_to_insertion(self.TIP, depth, pitch, yaw)
                np.testing.assert_allclose(
                    insertion_to_tip(entry, depth, pitch, yaw), self.TIP, atol=1e-9,
                )

    def test_insertion_lies_on_the_shank(self):
        # The entry point *is* "the point depth µm up the shank", so it must
        # coincide with the electrode-array math for every shank orientation.
        for pitch, yaw in POSES:
            for spin in SPINS:
                np.testing.assert_allclose(
                    tip_to_insertion(self.TIP, 3000.0, pitch, yaw),
                    probe_to_atlas([[0.0, 3000.0]], self.TIP, pitch, yaw, spin)[0],
                    atol=1e-9,
                )

    def test_depth_is_along_shank_not_dv(self):
        # Depth is manipulator travel down the shank, not a vertical drop:
        # at 60° of pitch a 1000 µm insertion only gains 500 µm of DV.
        delta = tip_to_insertion(self.TIP, 1000.0, 60, 0) - np.array(self.TIP)
        assert abs(abs(delta[2]) - 500.0) < 1e-6
        assert abs(np.linalg.norm(delta) - 1000.0) < 1e-6

    def test_no_singularity_at_ninety_degrees(self):
        # A horizontal probe is a legal pose; a vertical-depth
        # parameterisation would divide by cos(90°) here.
        entry = tip_to_insertion(self.TIP, 1000.0, 90, 0)
        assert np.all(np.isfinite(entry))
        np.testing.assert_allclose(
            insertion_to_tip(entry, 1000.0, 90, 0), self.TIP, atol=1e-9)

    def test_depth_recovers_from_two_points(self):
        for pitch, yaw in POSES:
            entry = tip_to_insertion(self.TIP, 2750.0, pitch, yaw)
            assert abs(insertion_depth_um(self.TIP, entry, pitch, yaw) - 2750.0) < 1e-6

    def test_depth_is_negative_when_tip_is_above_the_entry_point(self):
        # The probe has not reached that point yet. A norm-based
        # implementation would wrongly report a healthy positive depth.
        entry = np.array(self.TIP) + np.array([0.0, 0.0, 500.0])  # deeper than the tip
        assert insertion_depth_um(self.TIP, entry) == pytest.approx(-500.0)


class TestShankTipOffset:
    """Depth is manipulator travel of the *physical* shank tip, while
    ``tip_atlas`` is the lowest electrode — ``tip_length_um`` bridges them."""

    TIP = (5000.0, 2500.0, 3000.0)
    TIP_LEN = 209.0                       # NP 1.0; NP 2.0 is 206 µm

    def test_offset_shortens_the_electrode_reach(self):
        # Of 4000 µm of travel, the last 209 µm is inactive silicon, so the
        # lowest electrode only makes it 3791 µm below the surface.
        entry = tip_to_insertion(self.TIP, 4000.0, 0, 0, self.TIP_LEN)
        assert np.linalg.norm(entry - np.array(self.TIP)) == pytest.approx(3791.0)

    def test_zero_offset_reproduces_electrode_referenced_depth(self):
        for pitch, yaw in POSES:
            np.testing.assert_allclose(
                tip_to_insertion(self.TIP, 4000.0, pitch, yaw, 0.0),
                tip_to_insertion(self.TIP, 4000.0, pitch, yaw),
                atol=1e-9,
            )

    def test_round_trip_with_offset(self):
        for depth in (0.0, 209.0, 4000.0):
            for pitch, yaw in POSES:
                entry = tip_to_insertion(self.TIP, depth, pitch, yaw, self.TIP_LEN)
                np.testing.assert_allclose(
                    insertion_to_tip(entry, depth, pitch, yaw, self.TIP_LEN),
                    self.TIP, atol=1e-9,
                )

    def test_depth_recovers_with_offset(self):
        for pitch, yaw in POSES:
            entry = tip_to_insertion(self.TIP, 3500.0, pitch, yaw, self.TIP_LEN)
            got = insertion_depth_um(self.TIP, entry, pitch, yaw, self.TIP_LEN)
            assert got == pytest.approx(3500.0)

    def test_depth_equal_to_tip_length_puts_electrodes_at_the_surface(self):
        # Shank tip exactly tip_length deep → the lowest electrode is level
        # with the brain surface.
        entry = tip_to_insertion(self.TIP, self.TIP_LEN, 20, -10, self.TIP_LEN)
        np.testing.assert_allclose(entry, self.TIP, atol=1e-9)

    def test_shallower_than_tip_length_leaves_electrodes_above_the_surface(self):
        # Physically meaningful, not an error: the tip has entered but the
        # electrodes have not. The lowest electrode ends up dorsal of the
        # entry point, i.e. at smaller DV.
        entry = tip_to_insertion(self.TIP, 100.0, 0, 0, self.TIP_LEN)
        assert entry[2] > self.TIP[2], "entry should be deeper than the electrode"

    def test_probe_features_supply_a_tip_length_for_every_supported_probe(self):
        # The GUI reads this per part number; a missing or zero value would
        # silently fall back to electrode-referenced depth.
        from pixelmap.constants import PROBE_TYPE_MAP, WIRING_FILE_MAP
        from pixelmap.utils.probe_features import PROBE_FEATURES

        for probe_type in WIRING_FILE_MAP:
            for part_number in PROBE_TYPE_MAP[probe_type]:
                tip_len = PROBE_FEATURES[part_number]["tip_length_um"]
                assert tip_len > 0, f"{part_number} has no tip length"
                # Sanity band: every Neuropixels shank tip is ~0.2 mm.
                assert 150.0 < tip_len < 400.0, f"{part_number}: {tip_len}"


class TestBregmaRoundTrip:
    """atlas_to_bregma_um must exactly invert bregma_to_atlas_um."""

    def test_inverts_forward_transform(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            coords = tuple(rng.uniform(-6000, 6000, 3))
            bregma = tuple(rng.uniform(0, 9000, 3))
            # dv_squish=0 included on purpose: the forward guards it with a
            # truthiness test, so the inverse has to guard it identically.
            squish = float(rng.choice([1.0, 0.885, 0.5, 0.0]))
            tilt = float(rng.uniform(-30, 30))
            atlas = bregma_to_atlas_um(
                coords, bregma_um=bregma, dv_squish=squish, tilt_deg=tilt)
            back = atlas_to_bregma_um(
                atlas, bregma_um=bregma, dv_squish=squish, tilt_deg=tilt)
            np.testing.assert_allclose(back, coords, atol=1e-6)

    def test_ap_axis_flips(self):
        # Bregma-frame +AP is anterior, which is *decreasing* atlas AP.
        bregma = (5200.0, 5700.0, 440.0)
        out = atlas_to_bregma_um((4200.0, 5700.0, 440.0), bregma_um=bregma)
        assert out[0] == pytest.approx(1000.0)

    def test_display_offset_is_independent_of_the_landmark(self):
        # Why the GUI does not re-mirror when the landmark moves: with
        # squish/tilt neutral the display transform's linear part is
        # constant, so the tip→entry offset it shows depends only on the
        # pose. If real squish/tilt are ever folded in, this breaks — which
        # is exactly the signal to re-sync on those inputs.
        tip = np.array([6000.0, 5000.0, 4000.0])
        entry = tip + 3000.0 * probe_axis(15, -10)
        offsets = []
        for bregma in [(5200.0, 5700.0, 440.0), (100.0, 20.0, 7.0)]:
            t = np.array(atlas_to_bregma_um(tuple(tip), bregma_um=bregma))
            e = np.array(atlas_to_bregma_um(tuple(entry), bregma_um=bregma))
            offsets.append(e - t)
        np.testing.assert_allclose(offsets[0], offsets[1], atol=TOL)
