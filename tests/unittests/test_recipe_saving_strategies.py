# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the named saving-strategy factory functions defined inside the recipe files.

Each `process_*.py` recipe under `el_paso/recipes/` defines its own small
`<...>_strategy(...)` function(s) that wire the mission/satellite/instrument/data-standard
literals for that recipe into the right `ep.saving_strategies.*Strategy` class. The strategy
classes themselves already have their own read/write behavior covered under
`tests/unittests/saving_strategies/`; these tests just check each factory builds the right
class with the right attributes, and that overridable keyword-only arguments
(`file_format`, `data_standard`) actually take effect.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import el_paso as ep
import pytest
from el_paso.recipes.arase.process_arase_mepe import (
    arase_mepe_gfz_strategy,
    arase_mepe_h5_strategy,
    arase_mepe_netcdf_strategy,
)
from el_paso.recipes.arase.process_arase_pwe_densities import arase_pwe_densities_strategy
from el_paso.recipes.arase.process_arase_xep import arase_xep_strategy
from el_paso.recipes.arase.process_arase_xep_realtime import (
    arase_xep_realtime_gfz_strategy,
)
from el_paso.recipes.arase.process_arase_xep_realtime import (
    arase_xep_realtime_strategy as arase_xep_realtime_netcdf_strategy,
)
from el_paso.recipes.dmsp.process_dmsp_ssj_electrons import dmsp_ssj_electron_strategy
from el_paso.recipes.esa.process_ngrm_satellite import esa_ngrm_strategy
from el_paso.recipes.goes.process_goes_r_mps_high import (
    goes_r_mps_high_gfz_strategy,
    goes_r_mps_high_netcdf_strategy,
)
from el_paso.recipes.goes.process_goes_realtime import (
    goes_realtime_gfz_strategy,
    goes_realtime_netcdf_strategy,
)
from el_paso.recipes.gps.process_gps import gps_cxd_strategy
from el_paso.recipes.poes.process_poes_meped import poes_meped_strategy
from el_paso.recipes.poes.process_poes_ted import poes_ted_strategy
from el_paso.recipes.probav.process_ept_electron_fluxes import (
    probav_ept_electron_gfz_strategy,
    probav_ept_electron_netcdf_strategy,
)
from el_paso.recipes.probav.process_ept_proton_fluxes import (
    probav_ept_proton_gfz_strategy,
    probav_ept_proton_netcdf_strategy,
)
from el_paso.recipes.rbsp.process_rbsp_ect_combined import (
    rbsp_ect_combined_gfz_strategy,
    rbsp_ect_combined_netcdf_strategy,
)
from el_paso.recipes.rbsp.process_rbsp_emfisis_waves import rbsp_emfisis_waves_strategy
from el_paso.recipes.rbsp.process_rbsp_hope_electrons import (
    rbsp_hope_electron_gfz_strategy,
    rbsp_hope_electron_netcdf_strategy,
)
from el_paso.recipes.rbsp.process_rbsp_hope_protons import (
    rbsp_hope_proton_gfz_strategy,
    rbsp_hope_proton_netcdf_strategy,
)
from el_paso.recipes.rbsp.process_rbsp_mageis_electrons import rbsp_mageis_electron_strategy
from el_paso.recipes.rbsp.process_rbsp_mageis_protons import (
    rbsp_mageis_proton_gfz_strategy,
    rbsp_mageis_proton_netcdf_strategy,
)
from el_paso.recipes.rbsp.process_rbsp_rbspice_protons import (
    rbsp_rbspice_proton_gfz_strategy,
    rbsp_rbspice_proton_netcdf_strategy,
)
from el_paso.recipes.themis.process_themis_fft_waves import themis_fft_waves_strategy
from el_paso.recipes.themis.process_themis_scpot_density import themis_scpot_density_strategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from el_paso.saving_strategy import SavingStrategy


# Each case is (factory, extra_args, extra_kwargs, expected_cls, expected_attrs).
# `base_data_path` (tmp_path) is prepended by the test itself, since it isn't known at
# collection time.
CASES: list[
    tuple[Callable[..., SavingStrategy], tuple[Any, ...], dict[str, Any], type[SavingStrategy], dict[str, Any]]
] = [
    (
        arase_xep_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "xep", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        arase_xep_realtime_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "Arase", "satellite": "Arase", "instrument": "XEP", "mag_field": "T89"},
    ),
    (
        arase_xep_realtime_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "xep", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        arase_mepe_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "ARASE", "satellite": "arase", "instrument": "mepe", "mag_field": "T89"},
    ),
    (
        arase_mepe_h5_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "mepe", "mag_field": "T89", "file_format": ".h5"},
    ),
    (
        arase_mepe_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "mepe", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        arase_pwe_densities_strategy,
        ("T89",),
        {},
        ep.saving_strategies.MonthlyDensityStrategy,
        {"mission": "Arase", "satellite": "arase", "instrument": "PWE", "mag_field": "T89"},
    ),
    (
        themis_scpot_density_strategy,
        ("a", "T89"),
        {},
        ep.saving_strategies.MonthlyDensityStrategy,
        {"mission": "THEMIS", "satellite": "tha", "instrument": "SCPOT", "mag_field": "T89"},
    ),
    (
        dmsp_ssj_electron_strategy,
        ("T89", "f16"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "DMSP", "satellite": "f16", "instrument": "SSJ", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        esa_ngrm_strategy,
        ("T89", "S6-MF"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "ESA", "satellite": "s6-mf", "instrument": "ngrm", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        esa_ngrm_strategy,
        ("T89", "EDRS-C"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "ESA", "satellite": "edrs-c", "instrument": "ngrm", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        goes_r_mps_high_gfz_strategy,
        ("T89", "goes16"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "GOES", "satellite": "goes16", "instrument": "MAGED", "mag_field": "T89"},
    ),
    (
        goes_r_mps_high_netcdf_strategy,
        ("T89", "goes16"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "GOES", "satellite": "goes16", "instrument": "MAGED", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        goes_realtime_gfz_strategy,
        ("T89", "16"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "GOES", "satellite": "goes_16", "instrument": "mps-high", "mag_field": "T89"},
    ),
    (
        goes_realtime_netcdf_strategy,
        ("T89", "16"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {
            "mission": "GOES",
            "satellite": "goes_16",
            "instrument": "mps-high",
            "mag_field": "T89",
            "file_format": ".nc",
        },
    ),
    (
        gps_cxd_strategy,
        ("T89", "ns41"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "GPS", "satellite": "ns41", "instrument": "cxd", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        poes_ted_strategy,
        ("T89", "noaa19"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "POES", "satellite": "noaa19", "instrument": "TED", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        poes_meped_strategy,
        ("T89", "noaa19"),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "POES", "satellite": "noaa19", "instrument": "MEPED", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        probav_ept_electron_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "PROBAV", "satellite": "probav", "instrument": "ept", "mag_field": "T89"},
    ),
    (
        probav_ept_electron_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {"mission": "PROBAV", "satellite": "probav", "instrument": "ept", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        probav_ept_proton_gfz_strategy,
        ("T89",),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "PROBAV", "satellite": "probav", "instrument": "EPT-proton", "mag_field": "T89"},
    ),
    (
        probav_ept_proton_netcdf_strategy,
        ("T89",),
        {},
        ep.saving_strategies.DailyLEORBStrategy,
        {
            "mission": "PROBAV",
            "satellite": "probav",
            "instrument": "EPT-proton",
            "mag_field": "T89",
            "file_format": ".nc",
        },
    ),
    (
        rbsp_ect_combined_gfz_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "ect_combined", "mag_field": "T89"},
    ),
    (
        rbsp_ect_combined_netcdf_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {
            "mission": "RBSP",
            "satellite": "rbspa",
            "instrument": "ect_combined",
            "mag_field": "T89",
            "file_format": ".nc",
        },
    ),
    (
        rbsp_hope_electron_gfz_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "hope", "mag_field": "T89"},
    ),
    (
        rbsp_hope_electron_netcdf_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "hope", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rbsp_hope_proton_gfz_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "hope", "mag_field": "T89"},
    ),
    (
        rbsp_hope_proton_netcdf_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "hope", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rbsp_mageis_electron_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "mageis", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rbsp_mageis_proton_gfz_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "mageis", "mag_field": "T89"},
    ),
    (
        rbsp_mageis_proton_netcdf_strategy,
        ("T89", "b"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspb", "instrument": "mageis", "mag_field": "T89", "file_format": ".nc"},
    ),
    (
        rbsp_rbspice_proton_gfz_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.GFZStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "rbspice", "mag_field": "T89"},
    ),
    (
        rbsp_rbspice_proton_netcdf_strategy,
        ("T89", "a"),
        {},
        ep.saving_strategies.MonthlyRBStrategy,
        {"mission": "RBSP", "satellite": "rbspa", "instrument": "rbspice", "mag_field": "T89", "file_format": ".nc"},
    ),
]


def _case_id(factory: Callable[..., SavingStrategy], extra_kwargs: dict[str, Any]) -> str:
    suffix = f"[{','.join(f'{k}={v}' for k, v in extra_kwargs.items())}]" if extra_kwargs else ""
    return getattr(factory, "__qualname__", repr(factory)) + suffix


CASE_IDS = [_case_id(factory, kwargs) for factory, _args, kwargs, _cls, _attrs in CASES]

# The density strategies default to PRBEMStandard rather than GFZStandard. Mapped to the
# extra positional arguments their factory needs after `base_data_path`.
_PRBEM_DEFAULTING_FACTORIES: dict[Callable[..., SavingStrategy], tuple[Any, ...]] = {
    arase_pwe_densities_strategy: ("T89",),
    themis_scpot_density_strategy: ("a", "T89"),
}


@pytest.mark.basic
@pytest.mark.parametrize(
    ("factory", "extra_args", "extra_kwargs", "expected_cls", "expected_attrs"), CASES, ids=CASE_IDS
)
def test_strategy_factory_builds_expected_strategy(
    tmp_path: Path,
    factory: Callable[..., SavingStrategy],
    extra_args: tuple[Any, ...],
    extra_kwargs: dict[str, Any],
    expected_cls: type[SavingStrategy],
    expected_attrs: dict[str, Any],
) -> None:
    strategy = factory(tmp_path, *extra_args, **extra_kwargs)

    assert type(strategy) is expected_cls
    assert strategy.base_data_path == tmp_path

    for attr_name, expected_value in expected_attrs.items():
        assert getattr(strategy, attr_name) == expected_value, attr_name

    if factory in _PRBEM_DEFAULTING_FACTORIES:
        # The density strategies default to PRBEMStandard, not GFZStandard; see
        # test_density_strategies_default_to_prbem_standard below.
        assert isinstance(strategy.data_standard, ep.data_standards.PRBEMStandard)
    else:
        assert isinstance(strategy.data_standard, ep.data_standards.GFZStandard)


@pytest.mark.basic
@pytest.mark.parametrize(
    ("factory", "extra_args"), _PRBEM_DEFAULTING_FACTORIES.items(), ids=lambda f: getattr(f, "__name__", "")
)
def test_density_strategies_default_to_prbem_standard(
    tmp_path: Path, factory: Callable[..., SavingStrategy], extra_args: tuple[Any, ...]
) -> None:
    """The density strategies fall back to `PRBEMStandard()` when handed no data standard.

    This used to yield `self.data_standard is None`: the strategy assigned the fallback
    *before* calling `MonthlyRBStrategy.__init__`, which then re-assigned the attribute from
    the raw (unfallen-back) argument. Both density strategies now assign it after the
    `super().__init__` call, so the documented fallback actually takes effect.
    """
    strategy = factory(tmp_path, *extra_args)

    assert isinstance(strategy.data_standard, ep.data_standards.PRBEMStandard)


@pytest.mark.basic
@pytest.mark.parametrize("file_format", ["h5", "cdf", "mat"])
def test_file_format_override_takes_effect(tmp_path: Path, file_format: str) -> None:
    strategy = arase_xep_strategy(tmp_path, "T89", file_format=file_format)  # ty:ignore[invalid-argument-type]

    assert strategy.file_format == "." + file_format  # ty:ignore[unresolved-attribute]


def test_file_format_is_keyword_only(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        arase_xep_strategy(tmp_path, "T89", "nc")  # ty:ignore[too-many-positional-arguments]


@pytest.mark.basic
@pytest.mark.parametrize(
    "data_standard",
    [ep.data_standards.GFZStandard(), ep.data_standards.PRBEMStandard()],
    ids=lambda ds: type(ds).__name__,
)
def test_arase_mepe_data_standard_override_takes_effect(
    tmp_path: Path, data_standard: ep.typing.DataStandard[Any]
) -> None:
    strategy = arase_mepe_netcdf_strategy(tmp_path, "T89", data_standard)

    assert strategy.data_standard is data_standard


@pytest.mark.basic
def test_rbsp_emfisis_waves_strategy(tmp_path: Path) -> None:
    strategy = rbsp_emfisis_waves_strategy(tmp_path, "a")

    assert type(strategy) is ep.saving_strategies.DailyWaveStrategy
    assert strategy.mission == "RBSP"
    assert strategy.satellite == "rbspa"
    assert strategy.instrument == "EMFISIS"
    assert isinstance(strategy.data_standard, ep.data_standards.GFZStandard)


@pytest.mark.basic
def test_rbsp_emfisis_waves_strategy_data_standard_override(tmp_path: Path) -> None:
    prbem = ep.data_standards.PRBEMStandard()

    strategy = rbsp_emfisis_waves_strategy(tmp_path, "b", prbem)

    assert strategy.data_standard is prbem


@pytest.mark.basic
def test_themis_fft_waves_strategy(tmp_path: Path) -> None:
    strategy = themis_fft_waves_strategy(tmp_path, "a")

    assert type(strategy) is ep.saving_strategies.DailyWaveStrategy
    assert strategy.mission == "THEMIS"
    assert strategy.satellite == "tha"
    assert strategy.instrument == "FFT"
    assert isinstance(strategy.data_standard, ep.data_standards.GFZStandard)


@pytest.mark.basic
def test_themis_fft_waves_strategy_data_standard_override(tmp_path: Path) -> None:
    prbem = ep.data_standards.PRBEMStandard()

    strategy = themis_fft_waves_strategy(tmp_path, "d", prbem)

    assert strategy.data_standard is prbem


@pytest.mark.basic
def test_every_recipe_strategy_is_exported_from_its_mission_package() -> None:
    """Every `<...>_strategy` factory must be importable as `el_paso.recipes.<mission>.<name>`.

    Mirrors `test_every_recipe_is_exported_from_its_mission_package` in `test_recipe_cli.py`,
    but for the strategy factories rather than the `process_*` entry points. Regression test
    for https://github.com/GFZ/EL_PASO/issues/148, where none of the `<...>_strategy`
    functions were re-exported from their mission's `__init__.py`.
    """
    recipes_dir = Path(ep.recipes.__file__).parent

    for module_path in sorted(recipes_dir.glob("*/process_*.py")):
        mission = module_path.parent.name
        module_name = f"el_paso.recipes.{mission}.{module_path.stem}"
        module = importlib.import_module(module_name)
        mission_package = importlib.import_module(f"el_paso.recipes.{mission}")
        exported_names = getattr(mission_package, "__all__", ())

        strategy_factories = {
            name: obj
            for name, obj in vars(module).items()
            if name.endswith("_strategy") and inspect.isfunction(obj) and obj.__module__ == module_name
        }

        for name, factory in strategy_factories.items():
            assert any(getattr(mission_package, exported, None) is factory for exported in exported_names), (
                f"{module_name}.{name} is not exported (under any name) from el_paso/recipes/{mission}/__init__.py"
            )
