######################
## Global variables ##
######################

from pixelmap.utils.probe_features import (
    PROBE_FEATURES,
    PROBE_TYPE_MAP,
    LEGACY_PROBE_TYPE_MAP,
    LEGACY_INT_TO_IMRO_FORMAT,
    IMRO_FORMAT_TO_PROBE_TYPE,
)

# ---------------------------------------------------------------------------
# Wiring / position file map  (unchanged – keyed by internal probe type)
# ---------------------------------------------------------------------------
WIRING_FILE_MAP = {
    "1.0": ("1.0_positions.csv", "1.0_wiring.csv"),
    "2.0-1shank": ("2.0-1shank_positions.csv", "2.0-1shank_wiring.csv"),
    "2.0-4shanks": ("2.0-4shanks_positions.csv", "2.0-4shanks_wiring.csv"),
    "QuadBase": ("QuadBase_positions.csv", "QuadBase_wiring.csv"),
}

# ---------------------------------------------------------------------------
# Reference electrode definitions  (now keyed by IMRO format type)
# See https://billkarsh.github.io/SpikeGLX/help/imroTables/
# ---------------------------------------------------------------------------
REF_ELECTRODES = {
    # NP 1.0 family: {0=ext, 1=tip, [2..4]=on-shank-ref}
    "imro_np1000": {"External": 0, "Tip": 1},

    # NP 2.0 Phase-1 single-shank (type 21): {0=ext, 1=tip}
    "imro_np2000": {"External": 0, "Tip": 1},

    # NP 2.0 Phase-2B single-shank (types 2003/2004): {0=ext, 1=gnd, 2=tip}
    "imro_np2003": {"External": 0, "Ground": 1, "Tip": 2},

    # NP 2.0 Phase-1 four-shank (type 24):
    # {0=ext, 1..4=tip[0..3], 5..8=on-shank-0, ...}
    "imro_np2010": {
        "External": 0,
        "Tip shank 0": 1, "Tip shank 1": 2,
        "Tip shank 2": 3, "Tip shank 3": 4,
        "Join Tips": [1, 2, 3, 4] + [1] * 380,
    },

    # NP 2.0 Phase-2B four-shank (types 2013/2014):
    # {0=ext, 1=gnd, 2..5=tip[0..3]}
    "imro_np2013": {
        "External": 0, "Ground": 1,
        "Tip shank 0": 2, "Tip shank 1": 3,
        "Tip shank 2": 4, "Tip shank 3": 5,
        "Join Tips": [2, 3, 4, 5] + [2] * 380,
    },

    # NP 2.0 QuadBase (types 2020/2021): {0=ext, 1=gnd, 2=tip (same shank)}
    "imro_np2020": {"External": 0, "Ground": 1, "Tip": 2},
}

# ---------------------------------------------------------------------------
# Bank encoding  (keyed by internal probe type)
# ---------------------------------------------------------------------------
REF_BANKS = {
    "1.0":         {0: 0, 1: 1, 2: 2},
    "2.0-1shank":  {0: 0, 1: 2, 2: 4, 3: 8},  # bank_mask – powers of 2
    "2.0-4shanks": {0: 0, 1: 1, 2: 2, 3: 3},
    "QuadBase":    {0: 0, 1: 1, 2: 2, 3: 3},
}

# ---------------------------------------------------------------------------
# Preset lists  (unchanged)
# ---------------------------------------------------------------------------
SUPPORTED_1shank_PRESETS = [
    "Tip",
    "tip_b0_top_b1",
    "top_b0_tip_b1",
    "zigzag",
]

SUPPORTED_4shanks_PRESETS = [
    "tips_all",
    "tip_s0", "tip_s1", "tip_s2", "tip_s3",
    "tips_0_3", "tips_1_2",
    "tip_b0_top_b1_s0", "tip_b0_top_b1_s1",
    "tip_b0_top_b1_s2", "tip_b0_top_b1_s3",
    "top_b0_tip_b1_s0", "top_b0_tip_b1_s1",
    "top_b0_tip_b1_s2", "top_b0_tip_b1_s3",
    "tip_s0b0_top_s2b0", "tip_s2b0_top_s0b0",
    "tip_s1b0_top_s3b0", "tip_s3b0_top_s1b0",
    "gliding_0-3", "gliding_3-0",
    "zigzag_0", "zigzag_1", "zigzag_2", "zigzag_3",
]

SUPPORTED_QuadBase_PRESETS = [
    "tips_all",         # electrodes 0-383 (bank 0) on all 4 shanks — 1536 total
    "bank1_all",        # electrodes 384-767 (bank 1) on all 4 shanks — 1536 total
    "bank2_all",        # electrodes 768-1151 (bank 2) on all 4 shanks — 1536 total
]
