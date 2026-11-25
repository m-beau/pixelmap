"""Type definitions for the channelmap generator."""

from __future__ import annotations

from dataclasses import dataclass

# Note: frozen attributes (@dataclass(frozen=True)) allows hashability
# (thus usability as dictionnary keys or set elements)
@dataclass(frozen=True) 
class Electrode:
    """Neuropixels electrode electrode_id on shank shank_id."""

    shank_id: int
    electrode_id: int

