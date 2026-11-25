"""Region search: forgiving text matching, and a target point inside a region."""

from __future__ import annotations

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import search as search_module
from pixelmap.anatomy.search import (
    RegionEntry,
    region_span_along_shank,
    region_target_um,
    search_regions,
)
from pixelmap.anatomy.transform import AtlasWarp, ProbePlacement

ENTRIES = (
    RegionEntry(1, "VISp", "Primary visual area"),
    RegionEntry(2, "VISp1", "Primary visual area, layer 1"),
    RegionEntry(3, "VISpm", "posteromedial visual area"),
    RegionEntry(4, "CA1", "Field CA1"),
    RegionEntry(5, "HIP", "Hippocampal region"),
    RegionEntry(6, "CB", "Cerebellum"),
    RegionEntry(7, "MOp", "Primary motor area"),
    RegionEntry(8, "DG-sg", "Dentate gyrus, granule cell layer"),
    RegionEntry(9, "TH", "Thalamus"),
)


def _acronyms(query, limit=12):
    return [e.acronym for e in search_regions(query, ENTRIES, limit)]


class TestSearchRegions:
    def test_empty_query_matches_nothing(self):
        assert search_regions("", ENTRIES) == []
        assert search_regions("   ", ENTRIES) == []

    @pytest.mark.parametrize("query", ["VISp", "visp", " VisP "])
    def test_exact_acronym_ranks_first_whatever_the_case(self, query):
        found = _acronyms(query)
        assert found[0] == "VISp"
        assert set(found[1:3]) == {"VISp1", "VISpm"}

    def test_acronym_prefix(self):
        found = _acronyms("vis")
        assert found[0] == "VISp" and set(found) == {"VISp", "VISp1", "VISpm"}

    def test_plain_english_name(self):
        assert _acronyms("primary visual")[:2] == ["VISp", "VISp1"]
        assert _acronyms("thalamus") == ["TH"]

    def test_word_order_and_punctuation_do_not_matter(self):
        assert _acronyms("area visual primary")[0] == "VISp"
        assert _acronyms("granule dentate")[0] == "DG-sg"

    def test_half_typed_words(self):
        assert _acronyms("hippoc")[0] == "HIP"
        assert _acronyms("prim mot")[0] == "MOp"

    @pytest.mark.parametrize("query, expected", [
        ("cerebelum", "CB"), ("hipocampal", "HIP"), ("thalmus", "TH"), ("primery visual", "VISp"),
    ])
    def test_typos_are_forgiven(self, query, expected):
        assert _acronyms(query)[0] == expected

    def test_a_partly_matching_query_still_finds_the_region(self):
        """ "cortex" is in no name here, "visual" is."""
        assert set(_acronyms("visual cortex")) >= {"VISp", "VISpm"}

    def test_the_more_general_region_ranks_above_its_parts(self):
        assert _acronyms("primary visual area")[:2] == ["VISp", "VISp1"]

    def test_nonsense_matches_nothing(self):
        assert _acronyms("qqqqzzzz") == []

    def test_limit(self):
        assert len(_acronyms("a", limit=2)) <= 2


@pytest.fixture
def two_sided_atlas(monkeypatch):
    """A parent (id 10) with two leaves, painted as a C-shape in each hemisphere.

    The C wraps around a filler region (id 99), so the hemisphere centroid
    falls in the filler, not in the C — the case the snap exists for.
    """
    ann = np.zeros((20, 20, 40), dtype=np.int32)  # (AP, DV, ML) once canonical
    for ml0 in (2, 26):
        block = ann[4:16, 4:16, ml0:ml0 + 12]
        block[...] = 99
        block[:, :2, :] = 11      # top bar
        block[:, -2:, :] = 12     # bottom bar
        block[:2, :, :] = 11      # back bar
    ann[5, 5, 30] = 20            # a one-sided region

    class _Atlas:
        def __init__(self, name, **_kw):
            self.name, self.orientation = name, "asr"
            self.resolution = (25.0, 25.0, 25.0)
            self.annotation = ann
            self.structures = {
                10: {"id": 10, "acronym": "PAR", "name": "Parent", "structure_id_path": [10]},
                11: {"id": 11, "acronym": "L1", "name": "Leaf one", "structure_id_path": [10, 11]},
                12: {"id": 12, "acronym": "L2", "name": "Leaf two", "structure_id_path": [10, 12]},
                20: {"id": 20, "acronym": "ONE", "name": "One sided", "structure_id_path": [20]},
                30: {"id": 30, "acronym": "NONE", "name": "Unpainted", "structure_id_path": [30]},
                99: {"id": 99, "acronym": "FILL", "name": "Filler", "structure_id_path": [99]},
            }

    def _clear():
        atlas_module.release_atlas_memory()
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        search_module.region_catalog.cache_clear()
        search_module.region_target_um.cache_clear()

    _clear()
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _Atlas)
    yield ann
    _clear()


def _label_at(ann, point_um):
    ap, ml, dv = (round(c / 25.0) for c in point_um)
    return int(ann[ap, dv, ml])


class TestRegionTarget:
    def test_catalog_lists_every_structure(self, two_sided_atlas):
        labels = [e.label for e in search_module.region_catalog("two")]
        assert "PAR — Parent" in labels and len(labels) == 6

    def test_a_parent_region_resolves_through_its_leaves(self, two_sided_atlas):
        """The parent owns no voxel itself; the target must be in a leaf."""
        target = region_target_um("two", 10)
        assert _label_at(two_sided_atlas, target) in (11, 12)

    def test_target_is_inside_a_region_whose_centroid_is_not(self, two_sided_atlas):
        target = region_target_um("two", 11)
        assert _label_at(two_sided_atlas, target) == 11

    def test_the_requested_hemisphere_is_used(self, two_sided_atlas):
        midline_um = (40 - 1) / 2 * 25.0
        assert region_target_um("two", 10, right=True)[1] < midline_um
        assert region_target_um("two", 10, right=False)[1] > midline_um

    def test_a_one_sided_region_is_found_from_either_side(self, two_sided_atlas):
        expected = (5 * 25.0, 30 * 25.0, 5 * 25.0)
        assert region_target_um("two", 20, right=True) == expected
        assert region_target_um("two", 20, right=False) == expected

    def test_an_unpainted_region_has_no_target(self, two_sided_atlas):
        assert region_target_um("two", 30) is None

    def test_grid_sampling_is_a_regular_subset(self, two_sided_atlas):
        vol, _res = atlas_module.canonical_annotation("two")
        every = set(zip(*(a.tolist() for a in vol.find_label_voxels([99]))))
        ap, dv, ml = vol.sample_label_voxels([99], stride=3)
        grid = set(zip(ap.tolist(), dv.tolist(), ml.tolist()))
        assert grid == {v for v in every if all(c % 3 == 0 for c in v)}
        assert set(zip(*(a.tolist() for a in vol.sample_label_voxels([99], 1)))) == every

    def test_a_region_the_coarse_grid_misses_is_still_found(self, two_sided_atlas, monkeypatch):
        """One voxel, off the grid: the full-resolution fallback has to catch it."""
        monkeypatch.setattr(search_module, "_GRID_SAMPLES", 100)  # stride 6 on this volume
        search_module.region_target_um.cache_clear()
        assert region_target_um("two", 20) == (5 * 25.0, 30 * 25.0, 5 * 25.0)
        assert _label_at(two_sided_atlas, region_target_um("two", 11)) == 11


class TestRegionSpanAlongShank:
    """How much of a region shank 0 crosses, either side of the lowest electrode.

    In ``two_sided_atlas`` leaf 11 is a back bar (AP voxels 4–5, DV 4–15) plus
    a top bar (DV 4–5, AP 4–15); leaf 12 is the bottom bar (DV 14–15).
    """

    IN_BACK_BAR = (4 * 25.0, 5 * 25.0, 10 * 25.0)   # (AP, ML, DV) µm
    IN_TOP_BAR = (10 * 25.0, 5 * 25.0, 4 * 25.0)

    def test_a_vertical_shank_reaches_both_ends_of_the_region(self, two_sided_atlas):
        span = region_span_along_shank("two", 11, ProbePlacement(self.IN_BACK_BAR))
        # DV voxel 10 of 4–15: five voxels below, six above.
        assert span == (-125.0, 150.0)

    def test_the_span_follows_the_shank_axis(self, two_sided_atlas):
        """Pitched to horizontal, up the shank is +AP: along the top bar."""
        span = region_span_along_shank(
            "two", 11, ProbePlacement(self.IN_TOP_BAR, pitch_deg=90.0))
        np.testing.assert_allclose(span, (-150.0, 125.0))  # AP voxel 10 of 4–15

    def test_it_stops_where_the_shank_first_leaves_the_region(self, two_sided_atlas):
        """The parent is top bar + bottom bar; the filler between them ends the span."""
        span = region_span_along_shank("two", 10, ProbePlacement(self.IN_TOP_BAR))
        assert span == (-25.0, 0.0)  # DV voxels 4–5 only, not on to 14–15

    def test_a_tip_outside_the_region_has_no_span(self, two_sided_atlas):
        in_filler = (10 * 25.0, 5 * 25.0, 10 * 25.0)
        assert region_span_along_shank("two", 11, ProbePlacement(in_filler)) == (0.0, 0.0)
        assert region_span_along_shank("two", 30, ProbePlacement(self.IN_BACK_BAR)) == (0.0, 0.0)

    def test_the_span_is_measured_in_the_warped_brain(self, two_sided_atlas):
        """Squashed to half height, the same crossing is half as long."""
        warp = AtlasWarp((0.0, 0.0, 0.0), dv_squish=0.5)
        tip = tuple(warp.from_atlas(self.IN_BACK_BAR))
        span = region_span_along_shank("two", 11, ProbePlacement(tip, warp=warp))
        np.testing.assert_allclose(span, (-62.5, 75.0), atol=12.5)
        # ... and both ends, carried back into the atlas, are still in the region.
        ends = ProbePlacement(tip, warp=warp).to_atlas([[0.0, span[0]], [0.0, span[1]]])
        assert [_label_at(two_sided_atlas, e) for e in ends] == [11, 11]

    def test_reading_in_short_stretches_changes_nothing(self, two_sided_atlas, monkeypatch):
        monkeypatch.setattr(search_module, "_SPAN_CHUNK", 2)
        span = region_span_along_shank("two", 11, ProbePlacement(self.IN_BACK_BAR))
        assert span == (-125.0, 150.0)

