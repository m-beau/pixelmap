"""Regression tests for the package's public imports and documented API."""

import importlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import pixelmap
from pixelmap import backend, constants, types
from pixelmap.utils import imro

ROOT = Path(__file__).resolve().parents[1]
WIRING_DIR = ROOT / "pixelmap" / "wiring_maps"

# Supported public names and the module that owns each one. Adding or removing
# a name here is an intentional public API change.
PUBLIC_API = {
    backend: [
        "find_electrode_coordinates",
        "find_forbidden_electrodes",
        "find_selected_electrodes",
        "format_imro_string",
        "format_wiring_df",
        "generate_kilosort_channelmap_dict",
        "get_electrodes",
        "get_preset_candidates",
        "make_wiring_maps",
        "plot_probe_layout",
    ],
    constants: [
        "IMRO_FORMAT_TO_PROBE_TYPE",
        "LEGACY_INT_TO_IMRO_FORMAT",
        "LEGACY_PROBE_TYPE_MAP",
        "PROBE_FEATURES",
        "PROBE_TYPE_MAP",
        "REF_BANKS",
        "REF_ELECTRODES",
        "WIRING_FILE_MAP",
        "SUPPORTED_1shank_PRESETS",
        "SUPPORTED_4shanks_PRESETS",
        "SUPPORTED_QuadBase_PRESETS",
    ],
    types: ["Electrode"],
    imro: [
        "generate_imro_channelmap",
        "parse_imro_file",
        "parse_imro_list",
        "read_imro_file",
        "save_to_imro_file",
    ],
}
PUBLIC_NAMES = {name for names in PUBLIC_API.values() for name in names}
PUBLIC_SOURCES = [(module, name) for module, names in PUBLIC_API.items() for name in names]

# Names that ``from .backend import *`` used to leak as a side effect of
# backend's own imports. They must not be part of the package API.
INCIDENTAL_NAMES = [
    "np",
    "pd",
    "plt",
    "cm",
    "mcolors",
    "patches",
    "pickle",
    "Path",
    "PatchCollection",
    "PolyCollection",
    "pn",
]


def _star_import():
    namespace = {}
    exec("from pixelmap import *", namespace)  # noqa: S102 - exercise star-import semantics
    namespace.pop("__builtins__", None)
    return namespace


def _documented_files():
    paths = [
        path
        for path in (ROOT / "docs").rglob("*")
        if path.suffix in {".md", ".rst", ".py"} and "_build" not in path.parts
    ]
    paths.extend(ROOT.glob("README*"))
    return paths


def test_all_lists_exactly_the_supported_api():
    assert set(pixelmap.__all__) == PUBLIC_NAMES
    assert len(pixelmap.__all__) == len(set(pixelmap.__all__)), "duplicate names in __all__"


@pytest.mark.parametrize(("module", "name"), PUBLIC_SOURCES, ids=lambda x: getattr(x, "__name__", x))
def test_public_name_is_the_defining_object(module, name):
    # Re-exports must be the very same objects, not copies, so that e.g. mutating
    # pixelmap.PROBE_TYPE_MAP is seen by backend code and isinstance checks work.
    assert getattr(pixelmap, name) is getattr(module, name)


def test_legacy_probe_type_map_alias_is_preserved():
    assert pixelmap.LEGACY_PROBE_TYPE_MAP is constants.LEGACY_PROBE_TYPE_MAP
    assert pixelmap.LEGACY_PROBE_TYPE_MAP, "legacy alias map must not be empty"


def test_star_import_exports_exactly_all():
    namespace = _star_import()
    assert set(namespace) == set(pixelmap.__all__)
    for name, value in namespace.items():
        assert value is getattr(pixelmap, name)


def test_star_import_excludes_private_and_dunder_names():
    namespace = _star_import()
    assert not [name for name in namespace if name.startswith("_")]


@pytest.mark.parametrize("name", INCIDENTAL_NAMES)
def test_incidental_imports_are_not_public(name):
    assert name not in pixelmap.__all__
    assert not hasattr(pixelmap, name)
    assert name not in _star_import()


def test_submodules_remain_importable_but_not_star_exported():
    for module_name in ("backend", "constants", "types", "utils"):
        module = importlib.import_module(f"pixelmap.{module_name}")
        assert getattr(pixelmap, module_name) is module
        assert module_name not in pixelmap.__all__


def test_package_version_remains_accessible():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pixelmap.__version__)
    assert "__version__" not in pixelmap.__all__


def test_documented_api_names_resolve():
    references = set()
    for path in _documented_files():
        # Exclude URLs and paths such as github.com/m-beau/pixelmap.git.
        for name in re.findall(r"(?<![\w/.-])(?:cmg|pixelmap)\.(\w+)", path.read_text()):
            references.add(name)
            assert hasattr(pixelmap, name), f"{path.relative_to(ROOT)} references missing {name}"
    # Guard against the regex silently matching nothing.
    assert {"generate_imro_channelmap", "save_to_imro_file", "read_imro_file"} <= references


def test_documented_autodoc_targets_resolve():
    text = (ROOT / "docs" / "api.md").read_text()
    targets = []
    module = None
    for line in text.splitlines():
        if match := re.match(r"\.\. module:: ([\w.]+)", line):
            module = importlib.import_module(match.group(1))
        elif match := re.match(r"\.\. auto(?:function|class):: (\w+)", line):
            targets.append(match.group(1))
            assert hasattr(module, match.group(1)), f"docs/api.md autodoc target {match.group(1)}"
    assert targets


def test_documented_constants_are_public():
    text = (ROOT / "docs" / "api.md").read_text()
    section = text.split("## Constants", 1)[1].split("\n## ", 1)[0]
    names = re.findall(r"^`(\w+)`$", section, flags=re.MULTILINE)
    assert names
    for name in names:
        assert name in pixelmap.__all__, f"docs/api.md documents unexported constant {name}"


def test_notebook_imports_resolve():
    # The example notebook uses ``from pixelmap import ...``; it must keep working.
    notebook = json.loads((ROOT / "generate_channel_maps.ipynb").read_text())
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    imported = set()
    for names in re.findall(r"^from pixelmap import ([^*\n]+)$", source, flags=re.MULTILINE):
        imported.update(name.strip() for name in names.split(","))
    assert imported
    assert imported <= set(pixelmap.__all__)


@pytest.mark.parametrize(
    ("probe_type", "preset"),
    [("1.0", "Tip"), ("2.0-4shanks", "tips_all")],
)
def test_documented_workflow_round_trips(tmp_path, probe_type, preset):
    """Run the README/API-docs example using only top-level names."""
    imro_list = pixelmap.generate_imro_channelmap(
        probe_type=probe_type,
        layout_preset=preset,
        wiring_file=str(WIRING_DIR / f"{probe_type}_wiring.csv"),
    )
    path = tmp_path / "my_channelmap.imro"
    pixelmap.save_to_imro_file(imro_list, str(path))

    reread = pixelmap.read_imro_file(str(path))
    assert reread == imro_list
    selected, parsed_probe_type, *_ = pixelmap.parse_imro_list(reread)
    assert parsed_probe_type == probe_type
    assert len(selected) == 384
    assert len(np.unique(selected, axis=0)) == 384


def test_importing_the_package_does_not_load_the_gui_stack():
    """Script/API users shouldn't pay ~0.7 s to import Panel and Bokeh."""
    import subprocess
    import sys

    code = "import sys, pixelmap; print(sorted({'panel', 'bokeh'} & set(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
