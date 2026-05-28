"""Tests for the undo/redo stack on the Electrodes selection state."""

import pytest

from pixelmap.gui.gui import Electrodes
from pixelmap.types import Electrode


def _two_shank_wiring():
    """A minimal wiring map: 6 electrodes, no conflicts.

    Conflict-free keeps the tests focused on stack mechanics rather than
    re-derivation of available/unavailable; the conflict-aware path is
    covered separately by test_unavailable_set_re_derived_on_undo.
    """
    return {
        Electrode(0, 0): set(),
        Electrode(0, 1): set(),
        Electrode(0, 2): set(),
        Electrode(1, 0): set(),
        Electrode(1, 1): set(),
        Electrode(1, 2): set(),
    }


def _make_electrodes():
    return Electrodes(wiring_map=_two_shank_wiring(), n_maximum_electrodes=10)


class TestUndoRedoBasics:
    def test_initial_state_has_no_history(self):
        e = _make_electrodes()
        assert not e.can_undo()
        assert not e.can_redo()
        assert e.undo() is False
        assert e.redo() is False

    def test_single_undo_restores_previous_selection(self):
        e = _make_electrodes()
        e.select(Electrode(0, 0))

        e.push_undo_snapshot()
        e.select(Electrode(0, 1))
        assert e.selected == {Electrode(0, 0), Electrode(0, 1)}

        assert e.undo() is True
        assert e.selected == {Electrode(0, 0)}
        assert e.can_redo()

    def test_redo_restores_undone_selection(self):
        e = _make_electrodes()
        e.push_undo_snapshot()
        e.select(Electrode(0, 0))
        e.undo()
        assert e.selected == set()

        assert e.redo() is True
        assert e.selected == {Electrode(0, 0)}
        assert not e.can_redo()

    def test_new_snapshot_clears_redo_stack(self):
        e = _make_electrodes()
        e.push_undo_snapshot()
        e.select(Electrode(0, 0))
        e.undo()
        assert e.can_redo()

        # Branching off after an undo invalidates the redo branch.
        e.push_undo_snapshot()
        e.select(Electrode(0, 1))
        assert not e.can_redo()

    def test_undo_after_clear_selection(self):
        e = _make_electrodes()
        e.select(Electrode(0, 0))
        e.select(Electrode(1, 2))

        e.push_undo_snapshot()
        e.clear_selection()
        assert e.selected == set()

        e.undo()
        assert e.selected == {Electrode(0, 0), Electrode(1, 2)}


class TestStackBoundedness:
    def test_history_is_capped(self):
        e = _make_electrodes()
        e._max_history = 3

        for i in range(10):
            e.push_undo_snapshot()
        assert len(e._undo_stack) == 3


class TestWiringConflictRestoration:
    def test_unavailable_set_re_derived_on_undo(self):
        """When restoring a snapshot, conflicts must be recomputed."""
        wiring = {
            Electrode(0, 0): {Electrode(0, 1)},   # selecting 0 forbids 1
            Electrode(0, 1): {Electrode(0, 0)},
            Electrode(0, 2): set(),
        }
        e = Electrodes(wiring_map=wiring, n_maximum_electrodes=10)

        e.select(Electrode(0, 0))
        assert Electrode(0, 1) in e.unavailable
        assert Electrode(0, 1) not in e.available

        e.push_undo_snapshot()
        e.deselect(Electrode(0, 0))
        # After deselect, the conflict is gone.
        assert Electrode(0, 1) in e.available
        assert e.unavailable == set()

        e.undo()
        assert e.selected == {Electrode(0, 0)}
        # The conflict must have been re-derived from the wiring map.
        assert Electrode(0, 1) in e.unavailable
        assert Electrode(0, 1) not in e.available
