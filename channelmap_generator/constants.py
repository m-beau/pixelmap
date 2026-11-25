######################
## Global variables ##
######################

# Probe type to SpikeGLX type mapping
PROBE_TYPE_MAP = {
    "1.0": [0, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300],
    "2.0-1shank": [21, 2003, 2004],
    "2.0-4shanks": [24, 2013, 2014],
    # "quad": [2020, 2021],
    # "NXT": []
}

PROBE_N = { # N: number of physical electrodes; n: number of ACDs or channels
    "1.0": {'N': 960, 'n':384, 'n_per_shank':384},
    "2.0-1shank": {'N': 1280, 'n':384, 'n_per_shank':384},
    "2.0-4shanks": {'N': 5120, 'n':384, 'n_per_shank':384},
    # "NXT": {'N': 5120, 'n':1536, 'n_per_shank':908} # NXT - will apparently have a limit per shank
}

WIRING_FILE_MAP = {
    "1.0": ("1.0_positions.csv", "1.0_wiring.csv"),
    "2.0-1shank": ("2.0-1shank_positions.csv", "2.0-1shank_wiring.csv"),
    "2.0-4shanks": ("2.0-4shanks_positions.csv", "2.0-4shanks_wiring.csv"),
    # "NXT": ("2.0-4shanks_positions.csv", "2.0-4shanks_wiring.csv"),
}

# Reference electrode definitions
# from https://billkarsh.github.io/SpikeGLX/help/imroTables/
REF_ELECTRODES = {
    21: {'External':0, 'Tip':1}, # {0=ext, 1=tip, [2..5]=on-shnk-ref}
    2003: {'External':0, 'Ground':1, 'Tip':2}, # {0=ext, 1=gnd, 2=tip}
    2004: {'External':0, 'Ground':1, 'Tip':2}, # {0=ext, 1=gnd, 2=tip}
    24: {'External':0, 'Tip shank 0':1, 'Tip shank 1':2, 'Tip shank 2':3, 'Tip shank 3':4, 'Join Tips':[1, 2, 3, 4] + [1] * 380}, # {0=ext, [1..4]=tip[0..3], [5..8]=on-shnk-0, [9..12]=on-shnk-1, [13..16]=on-shnk-2, [17..20]=on-shnk-3}
    2013: {'External':0, 'Ground':1, 'Tip shank 0':2, 'Tip shank 1':3, 'Tip shank 2':4, 'Tip shank 3':5, 'Join Tips':[2, 3, 4, 5] + [2] * 380}, # {0=ext, 1=gnd, [2..5]=tip[0..3]}
    2014: {'External':0, 'Ground':1, 'Tip shank 0':2, 'Tip shank 1':3, 'Tip shank 2':4, 'Tip shank 3':5, 'Join Tips':[2, 3, 4, 5] + [2] * 380}, # {0=ext, 1=gnd, [2..5]=tip[0..3]}
    # 2020: {'External':0, 'Ground':1, 'Tip':2}, # {0=ext, 1=gnd, 2=tip on same shank as electrode}.
    # 2021: {'External':0, 'Ground':1, 'Tip':2}, # {0=ext, 1=gnd, 2=tip on same shank as electrode}.
    # NXT subtypes!
}
REF_ELECTRODES = {tp: {'External':0, 'Tip':1} for tp in PROBE_TYPE_MAP['1.0']} | REF_ELECTRODES # {0=ext, 1=tip, [2..4]=on-shnk-ref}. The on-shnk ref electrodes are {192,576,960}.

REF_BANKS = {
    "1.0": {0:0, 1:1, 2:2},
    "2.0-1shank": {0:0, 1:2, 2:4, 3:8}, # powers of 2...!
    "2.0-4shanks": {0:0, 1:1, 2:2, 3:3},
    # "NXT": {0:0, 1:1, 2:2, 3:3}
}

SUPPORTED_1shank_PRESETS = [
    # 1 shank presets
    "Tip",
    "tip_b0_top_b1",
    "top_b0_tip_b1",
    "zigzag",
]

SUPPORTED_4shanks_PRESETS = [
    # 4 shanks presets
    "tips_all",
    "tip_s0",
    "tip_s1",
    "tip_s2",
    "tip_s3",
    "tips_0_3",
    "tips_1_2",
    "tip_b0_top_b1_s0",
    "tip_b0_top_b1_s1",
    "tip_b0_top_b1_s2",
    "tip_b0_top_b1_s3",
    "top_b0_tip_b1_s0",
    "top_b0_tip_b1_s1",
    "top_b0_tip_b1_s2",
    "top_b0_tip_b1_s3",
    "tip_s0b0_top_s2b0",
    "tip_s2b0_top_s0b0",
    "tip_s1b0_top_s3b0",
    "tip_s3b0_top_s1b0",
    "gliding_0-3",
    "gliding_3-0",
    "zigzag_0",
    "zigzag_1",
    "zigzag_2",
    "zigzag_3"
]

