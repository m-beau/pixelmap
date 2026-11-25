"""Find an atlas region by acronym or name, and a point inside it to aim at.

Two halves, both free of GUI code:

* :func:`search_regions` — forgiving text search over an atlas's structure
  list, by acronym (``VISp``) or plain-English name (``primary visual``),
  tolerant of typos, partial words and word order.
* :func:`region_target_um` — a representative point *inside* a region, in
  canonical atlas ``(AP, ML, DV)`` µm, for moving a probe tip onto it.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from pixelmap.anatomy.atlas import canonical_annotation, get_atlas

# region_target_um: size of the coarse sampling grid, and how few hits on it
# mean "too small a region to trust the grid".
_GRID_SAMPLES = 4_000_000
_MIN_HITS = 200


@dataclass(frozen=True)
class RegionEntry:
    """One searchable structure of an atlas."""

    atlas_id: int
    acronym: str
    name: str

    @property
    def label(self) -> str:
        return f"{self.acronym} — {self.name}" if self.name else self.acronym


@functools.lru_cache(maxsize=4)
def region_catalog(atlas_name: str) -> tuple[RegionEntry, ...]:
    """Every structure of an atlas, parents included.

    Needs the atlas's metadata, which on a cold cache means a download: gate
    on :func:`pixelmap.anatomy.atlas.is_downloaded`. Holds only small tuples,
    never the atlas object (see ``_region_info_from_id`` for why that matters).
    """
    entries = []
    for key, st in get_atlas(atlas_name).structures.items():
        try:
            region_id = int(st.get("id", key))
        except (TypeError, ValueError):
            continue
        entries.append(RegionEntry(
            atlas_id=region_id,
            acronym=str(st.get("acronym", f"id{region_id}")),
            name=str(st.get("name", "")),
        ))
    return tuple(entries)


def _normalise(text: str) -> str:
    """Lower-case, with punctuation flattened to single spaces."""
    return re.sub(r"[^a-z0-9/]+", " ", text.lower()).strip()


def _token_similarity(token: str, words: list[str]) -> float:
    """How well one query word matches the best word of a name, in ``[0, 1]``.

    A query word is usually a *prefix in progress* ("hippoc"), so it is
    compared against each name word cut to its own length (plus one, to let a
    dropped letter through) as well as against the whole word.
    """
    best = 0.0
    for word in words:
        if word.startswith(token):
            return 1.0
        if token in word:
            best = max(best, 0.9)
            continue
        head = word[: len(token) + 1]
        best = max(best,
                   SequenceMatcher(None, token, head).ratio(),
                   SequenceMatcher(None, token, word).ratio())
    return best


def _exact_score(query: str, tokens: list[str], entry: RegionEntry) -> float:
    """Relevance of ``entry`` when the query matches it literally; else 0."""
    acronym, name = _normalise(entry.acronym), _normalise(entry.name)
    brevity = 1.0 / (1.0 + len(name))  # among equals, the more general region
    if query == acronym:
        return 100.0
    if query == name:
        return 98.0
    if acronym.startswith(query):
        return 90.0 - 0.1 * (len(acronym) - len(query))
    words = name.split()
    if all(any(w.startswith(t) for w in words) for t in tokens):
        return 82.0 + brevity
    if all(t in name for t in tokens):
        return 78.0 + brevity
    if len(query) >= 2 and query in acronym:
        return 70.0 - 0.1 * len(acronym)
    return 0.0


def _fuzzy_score(query: str, tokens: list[str], entry: RegionEntry) -> float:
    """Relevance of ``entry`` allowing typos and partly matching queries; else 0."""
    if len(query) < 3:  # too short to tell a typo from another word
        return 0.0
    acronym, name = _normalise(entry.acronym), _normalise(entry.name)
    brevity = 1.0 / (1.0 + len(name))
    words = name.split()
    by_name = 0.0
    if words:
        by_name = sum(_token_similarity(t, words) for t in tokens) / len(tokens)
    by_acronym = SequenceMatcher(None, query, acronym).ratio() if len(tokens) == 1 else 0.0
    if by_name >= 0.6 and by_name >= by_acronym:
        return 60.0 * by_name + brevity
    if by_acronym >= 0.75:
        return 55.0 * by_acronym
    return 0.0


def search_regions(query: str, entries, limit: int = 12) -> list[RegionEntry]:
    """The entries best matching ``query``, most relevant first.

    Matches the acronym or the name, case-insensitively: exact and prefix
    matches rank first, then names containing every query word (in any
    order), then approximate matches — a misspelt or half-typed word, or a
    query of which only some words appear in the name.
    """
    normalised = _normalise(query or "")
    if not normalised:
        return []
    tokens = normalised.split()
    scored, rest = [], []
    for entry in entries:
        score = _exact_score(normalised, tokens, entry)
        if score > 0.0:
            scored.append((score, entry))
        else:
            rest.append(entry)
    if len(scored) < limit:
        # Approximate matching is the slow part (a sequence alignment per
        # word), so it only runs when literal matches leave room in the list.
        for entry in rest:
            score = _fuzzy_score(normalised, tokens, entry)
            if score > 0.0:
                scored.append((score, entry))
    scored.sort(key=lambda se: (-se[0], len(se[1].name), se[1].acronym))
    return [entry for _score_, entry in scored[:limit]]


def _region_label_ids(atlas, region_id: int) -> list[int]:
    """Annotation labels belonging to a region: itself and every descendant.

    Parent structures (``Isocortex``, ``VISp``) own no voxels of their own in
    most atlases — the annotation is painted with their leaves.
    """
    ids = {int(region_id)}
    for key, st in atlas.structures.items():
        path = st.get("structure_id_path")
        if path is not None and region_id in path:
            ids.add(int(st.get("id", key)))
    return sorted(ids)


@functools.lru_cache(maxsize=256)
def region_target_um(
    atlas_name: str, region_id: int, right: bool = True
) -> tuple[float, float, float] | None:
    """A point inside a region to aim a probe tip at, as atlas ``(AP, ML, DV)`` µm.

    The centre of mass of the region's voxels in **one hemisphere**, snapped
    to the nearest voxel that really belongs to the region. Both steps matter:
    the centroid of a bilateral structure sits on the midline, outside it, and
    the centroid of a curved one (hippocampus, a cortical sheet) can fall in
    the neighbour it wraps around.

    Args:
        atlas_name: brainglobe atlas identifier.
        region_id: structure id; its descendants are included.
        right: preferred hemisphere (canonical ML counts from the right, so
            this is the low-ML half). If the region lies entirely in the
            other one, that one is used.

    Returns ``None`` if the atlas paints no voxel with this region.

    Slow: scans every voxel of the annotation (≈1 s for a 25 µm atlas, several
    at 10 µm), so call it from a worker thread. Memoised.
    """
    atlas = get_atlas(atlas_name)
    ann, res = canonical_annotation(atlas_name)  # indexed (AP, DV, ML)
    ids = _region_label_ids(atlas, region_id)
    # A coarse grid first: a few million samples locate any sizeable region,
    # where testing every voxel of a 10 µm atlas costs tens of seconds and
    # gigabytes. Too few hits means a small region — then every voxel is
    # needed, and affordable.
    stride = max(1, int(np.ceil((np.prod(ann.shape) / _GRID_SAMPLES) ** (1.0 / 3.0))))
    ap, dv, ml = ann.sample_label_voxels(ids, stride)
    if ap.size < _MIN_HITS and stride > 1:
        ap, dv, ml = ann.sample_label_voxels(ids, 1)
    if ap.size == 0:
        return None

    midline = (ann.shape[2] - 1) / 2.0
    on_right = ml <= midline
    side = on_right if right else ~on_right
    if not side.any():
        side = ~side
    ap, dv, ml = ap[side], dv[side], ml[side]

    centre = (ap.mean(), dv.mean(), ml.mean())
    nearest = int(np.argmin(
        (ap - centre[0]) ** 2 + (dv - centre[1]) ** 2 + (ml - centre[2]) ** 2
    ))
    return (float(ap[nearest] * res[0]), float(ml[nearest] * res[2]), float(dv[nearest] * res[1]))
