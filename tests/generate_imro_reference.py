"""Freeze 100 random full-readout IMROs using an unmodified dev checkout.

Run from the repository root: PYTHONPATH=. python tests/generate_imro_reference.py OUTPUT_DIR
OUTPUT_DIR must not exist, to prevent accidentally replacing the reference oracle.
"""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from pixelmap.constants import PROBE_FEATURES, PROBE_TYPE_MAP, REF_ELECTRODES, WIRING_FILE_MAP
from pixelmap.utils.imro import generate_imro_channelmap, save_to_imro_file

SEED = 20261029
ROOT = Path(__file__).resolve().parents[1]


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    cases = []
    reference_counts = {}
    for group, (_, wiring_name) in WIRING_FILE_MAP.items():
        wiring_file = ROOT / "pixelmap" / "wiring_maps" / wiring_name
        with wiring_file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        # Each CSV row is one readout channel. Choosing exactly one nonempty
        # cell per row gives full readout without ADC conflicts, also for QuadBase.
        choices = []
        for row in rows:
            candidates = []
            for column, value in row.items():
                if column != "channel" and value:
                    shank = int(column.split("-")[0].removeprefix("shank"))
                    candidates.append([shank, int(float(value))])
            choices.append(candidates)
        subtypes = PROBE_TYPE_MAP[group]
        for index in range(25):
            subtype = subtypes[index % len(subtypes)]
            features = PROBE_FEATURES[subtype]
            references = list(REF_ELECTRODES[features["IMRO_format"]])
            # Cover every reference for each wire format, including Join Tips.
            fmt = features["IMRO_format"]
            count = reference_counts.get(fmt, 0)
            reference = references[count % len(references)]
            reference_counts[fmt] = count + 1
            electrodes = [rng.choice(candidates) for candidates in choices]
            rng.shuffle(electrodes)  # Exercise sorting and input-order coordinate lookup.
            case = {
                "id": f"{group}-{index:02d}",
                "probe_type": group,
                "probe_subtype": subtype,
                "reference_id": reference,
                "ap_gain": rng.choice([50, 125, 250, 500, 1000, 1500, 2000, 3000]),
                "lf_gain": rng.choice([50, 125, 250, 500, 1000, 1500, 2000, 3000]),
                "hp_filter": index % 2,
                "electrodes": electrodes,
            }
            assert len(electrodes) == features["n_readouts_total"]
            imro = generate_imro_channelmap(
                **{
                    key: case[key]
                    for key in (
                        "probe_type",
                        "probe_subtype",
                        "reference_id",
                        "ap_gain",
                        "lf_gain",
                        "hp_filter",
                    )
                },
                custom_electrodes=np.array(electrodes),
                wiring_file=str(wiring_file),
            )
            save_to_imro_file(imro, str(output / f"{case['id']}.imro"))
            cases.append(case)
    manifest = {
        "seed": SEED,
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cases": cases,
    }
    # One case per line keeps this large numeric fixture compact and reviewable.
    with (output / "manifest.json").open("w") as stream:
        stream.write('{\n  "seed": ' + str(SEED) + ',\n  "baseline_commit": ')
        stream.write(json.dumps(manifest["baseline_commit"]) + ',\n  "cases": [\n')
        stream.write(",\n".join("    " + json.dumps(case) for case in cases))
        stream.write("\n  ]\n}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
