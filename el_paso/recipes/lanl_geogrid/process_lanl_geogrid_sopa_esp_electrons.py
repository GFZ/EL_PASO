# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso.recipes.lanl_geogrid import LANLGeoGridSatellite
from el_paso.recipes.lanl_geogrid._lanl_geogrid_extraction import extract_lanl_geogrid_ascii
from el_paso.recipes.solar_wind_indices.process_solar_wind_indices import process_solar_wind_indices
from el_paso.utils import timed_function


class _MonthlySopaEspElectronStrategy(ep.saving_strategies.MonthlyRBStrategy):
    """Monthly strategy that also saves the measured spin-averaged electron flux (FEDU_SA).

    It is not part of the shared monthly variable list, since missing variables are filled with empty
    placeholders there, which would end up in the output of every other recipe.
    """

    def _get_output_file_entries(self) -> list[ep.typing.InternalName | ep.typing.OneOf]:
        return [*super()._get_output_file_entries(), "FEDU_SA"]


def lanl_geogrid_sopa_esp_electron_strategy(
    processed_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    satellite: LANLGeoGridSatellite,
) -> ep.SavingStrategy:
    """Monthly GFZ-standard saving strategy for LANL GeoGrid SOPA/ESP electrons.

    Args:
        processed_data_path (str | Path): Base directory of the processed data.
        mag_field (ep.typing.MagneticFieldLiteral): Magnetic field model the data was processed with.
        satellite (LANLGeoGridSatellite): Geosynchronous longitude slot the data belongs to.

    Returns:
        ep.SavingStrategy: A monthly netCDF strategy using the GFZ data standard.
    """
    return _MonthlySopaEspElectronStrategy(
        Path(processed_data_path),
        "LANL_GeoGrid",
        f"geogrid_{satellite}",
        "sopa-esp-electrons",
        mag_field,
        data_standard=ep.data_standards.GFZStandard(),
        file_format=".nc",
    )


@timed_function("process_lanl_geogrid_sopa_esp_electrons")
def process_lanl_geogrid_sopa_esp_electrons(
    start_time: datetime,
    end_time: datetime,
    satellite: LANLGeoGridSatellite = "010",
    mag_field: ep.typing.MagneticFieldLiteral = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=1),
    num_cores: int = 16,
    save_strategy: Literal["netcdf"] = "netcdf",
    *,
    skip_existing: bool = True,
    calculate_Lstar: bool = True,
    save_indices_sw: bool = False,
    append: bool = False,
) -> None:
    """Process LANL GeoGrid SOPA and ESP spin-averaged electron fluxes into pitch-angle-resolved fluxes.

    Downloads the daily LowRes SOPA_ESP ASCII files for the given geosynchronous longitude slot
    from NOAA NCEI, and extracts the SOPA (10 channels) and ESP (8 channels, the ninth is a fill
    value and dropped) spin-averaged electron differential fluxes. Both are combined into a
    single spectrum (~50 keV to ~17 MeV). The data is binned to `bin_cadence`, the geographic
    position is converted to GEO coordinates, magnetic field quantities (B_Calc, B_Eq, MLT,
    R_Eq, L_m, L_star, Alpha_Eq, InvMu, InvK) are computed, and the spin-averaged flux is
    expanded into a pitch-angle-resolved flux (FEDU) via construct_pitch_angle_distribution
    (sine PAD shape, "spin_average" normalization). The electron phase space density is derived
    from FEDU, and the results are saved to monthly GFZ-standard files. The measured spin-averaged
    flux, without the synthetic pitch-angle distribution, is saved as well (`FEDU_SA`).

    Note: the fluxes are per steradian, so they are spin-averaged directional fluxes, not
    omnidirectional ones, and are not multiplied by 4 pi. The instrument provides no local
    pitch-angle information; the local pitch angles used here are a fixed synthetic grid
    (5 to 90 deg, step 5), not a measured quantity.

    Note: SC_RAD is labelled km in the file headers, but holds Earth radii.

    Args:
        start_time (datetime): Start of the time interval to process.
        end_time (datetime): End of the time interval to process.
        satellite (LANLGeoGridSatellite): The geosynchronous longitude slot to process, in
            degrees east (e.g. "010").
        mag_field (MagneticFieldLiteral): Magnetic field model used for the derived quantities.
        raw_data_path (str | Path): Directory where the raw downloaded data files are stored.
        processed_data_path (str | Path): Directory where the processed output files are saved.
        bin_cadence (timedelta): Time cadence used to bin the extracted variables (native ~10 s).
        num_cores (int): Number of CPU cores used for the magnetic field computations.
        save_strategy (Literal["netcdf"]): The saving strategy used to write the processed data.
            Only monthly netCDF files are supported.
        skip_existing (bool): If True, skip downloading files that already exist locally.
            Defaults to True.
        calculate_Lstar (bool): If True, also compute the L* magnetic field quantity. If False, the
            expensive drift-shell tracing is skipped and L* is not saved, while L_m and InvK are still
            computed. Defaults to True.
        save_indices_sw (bool): If True, also save all solar wind indices and parameters,
            independent of this recipe's own output. Defaults to False.
        append (bool): If True, append to existing output files instead of overwriting them.
            Defaults to False.
    """
    del save_strategy

    data_path_stem = f"{raw_data_path}/LANL_GeoGrid/{satellite}/YYYY/"
    url = "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_geogrid/data/LowRes/YYYY"
    file_name_stem = rf"YYYYMMDD_LANL_GeoGrid_{satellite}_LowRes_SOPA_ESP_v\d+\.\d+\.\d+\.txt"

    ep.download(
        start_time,
        end_time,
        save_path=data_path_stem,
        file_cadence="daily",
        download_url=url,
        file_name_stem=file_name_stem,
        skip_existing=skip_existing,
    )

    ep.exit_if_download_only()

    flux_unit = (u.cm**2 * u.s * u.sr * u.keV) ** (-1)

    extraction_infos = [
        ep.ExtractionInfo(result_key="Date", name_or_column="Date", unit=u.dimensionless_unscaled),
        ep.ExtractionInfo(result_key="UTC", name_or_column="UTC", unit=u.hour),
        ep.ExtractionInfo(result_key="SC_LAT", name_or_column="SC_LAT", unit=u.deg),
        ep.ExtractionInfo(result_key="SC_LON", name_or_column="SC_LON", unit=u.deg),
        ep.ExtractionInfo(result_key="SC_RAD", name_or_column="SC_RAD", unit=ep.units.RE),
        ep.ExtractionInfo(result_key="Sopa_E_Flux", name_or_column="Sopa_E_Flux", unit=flux_unit),
        ep.ExtractionInfo(result_key="Esp_E_Flux", name_or_column="Esp_E_Flux", unit=flux_unit),
        ep.ExtractionInfo(
            result_key="Sopa_E_Energy", name_or_column="Sopa_E_Energy", unit=u.keV, is_time_dependent=False
        ),
        ep.ExtractionInfo(
            result_key="Esp_E_Energy", name_or_column="Esp_E_Energy", unit=u.keV, is_time_dependent=False
        ),
    ]

    variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        file_cadence="daily",
        data_path=data_path_stem,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
        custom_extractors={".txt": extract_lanl_geogrid_ascii},
    )

    # combine SOPA and ESP into one spectrum; the last ESP channel has a fill value as energy
    esp_energy = variables["Esp_E_Energy"].get_data(u.keV)
    esp_valid_channels = np.isfinite(esp_energy)

    variables["FEDU_SA"] = ep.Variable(
        data=np.concatenate(
            (
                variables["Sopa_E_Flux"].get_data(flux_unit),
                variables["Esp_E_Flux"].get_data(flux_unit)[:, esp_valid_channels],
            ),
            axis=1,
        ),
        original_unit=flux_unit,
    )
    variables["Energy_FEDU"] = ep.Variable(
        data=np.concatenate((variables["Sopa_E_Energy"].get_data(u.keV), esp_energy[esp_valid_channels])),
        original_unit=u.keV,
    )

    for key in ("Sopa_E_Flux", "Esp_E_Flux", "Sopa_E_Energy", "Esp_E_Energy"):
        del variables[key]

    date = variables["Date"].get_data().astype(np.int64)
    unique_dates, date_idx = np.unique(date, return_inverse=True)
    day_start = np.array(
        [datetime.strptime(str(d), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() for d in unique_dates]
    )
    variables["Epoch"] = ep.Variable(
        data=day_start[date_idx] + variables["UTC"].get_data(u.s), original_unit=ep.units.posixtime
    )

    variables["FEDU_SA"].apply_thresholds_on_data(lower_threshold=0)

    # use (-180, 180] so slots near 0 deg do not wrap around inside a time bin
    lon = variables["SC_LON"].get_data(u.deg)
    variables["SC_LON"].set_data(np.where(lon > 180, lon - 360, lon), u.deg)

    time_bin_methods = {
        "FEDU_SA": ep.TimeBinMethod.NanMedian,
        "Energy_FEDU": ep.TimeBinMethod.Repeat,
        "SC_RAD": ep.TimeBinMethod.NanMedian,
        "SC_LAT": ep.TimeBinMethod.NanMedian,
        "SC_LON": ep.TimeBinMethod.NanMedian,
    }

    binned_time_var = ep.processing.bin_by_time(
        variables["Epoch"],
        variables,
        time_bin_methods,
        bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    # synthetic local pitch-angle grid, the instrument has no pitch-angle resolution
    pa_local_data = np.tile(np.arange(5, 91, 5), (len(binned_time_var.get_data()), 1)).astype(np.float64)
    variables["PA_local_FEDU"] = ep.Variable(data=pa_local_data, original_unit=u.deg)

    geo_spherical = np.vstack(
        (
            variables["SC_RAD"].get_data(ep.units.RE),
            variables["SC_LAT"].get_data(u.deg),
            variables["SC_LON"].get_data(u.deg),
        )
    ).T.astype(np.float64)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_var.get_data(ep.units.posixtime)]

    xgeo_arr = ep.processing.magnetic_field_utils.Coords().transform(
        datetimes,
        geo_spherical,
        ep.IRBEM_SYSAXIS_SPH,
        ep.IRBEM_SYSAXIS_GEO,
    )
    variables["xGEO"] = ep.Variable(data=xgeo_arr, original_unit=ep.units.RE)

    del variables["SC_LON"]
    del variables["SC_LAT"]

    variables_to_compute: ep.processing.VariableRequest = [
        ("B_Calc", mag_field),
        ("B_Eq", mag_field),
        ("MLT", mag_field),
        ("R_Eq", mag_field),
        ("L_m", mag_field),
        ("Alpha_Eq", mag_field),
        ("InvMu", mag_field),
        ("InvK", mag_field),
    ]

    if calculate_Lstar:
        variables_to_compute.append(("L_star", mag_field))

    irbem_options = ep.processing.magnetic_field_utils.IrbemOptions(
        lstar_quantity=ep.processing.magnetic_field_utils.LstarQuantity.LSTAR
        if calculate_Lstar
        else ep.processing.magnetic_field_utils.LstarQuantity.NONE,
    )

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time_var,
        xgeo_var=variables["xGEO"],
        energy_var=variables["Energy_FEDU"],
        pa_local_var=variables["PA_local_FEDU"],
        particle_species="electron",
        variables_to_compute=variables_to_compute,
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    FEDU_var = ep.processing.construct_pitch_angle_distribution(
        variables["FEDU_SA"],
        variables["PA_local_FEDU"],
        magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        flux_type="spin_average",
    )
    FEDU_var.apply_thresholds_on_data(lower_threshold=0)

    psd_var = ep.processing.compute_phase_space_density(FEDU_var, variables["Energy_FEDU"], particle_species="electron")

    variables_to_save: ep.typing.VariablesDict = {
        "Epoch": binned_time_var,
        "FEDU": FEDU_var,
        "FEDU_SA": variables["FEDU_SA"],
        "Energy_FEDU": variables["Energy_FEDU"],
        "Alpha": variables["PA_local_FEDU"],
        "R_Eq": magnetic_field_variables[f"R_Eq_{mag_field}"],
        "MLT": magnetic_field_variables[f"MLT_{mag_field}"],
        "B_Calc": magnetic_field_variables[f"B_Calc_{mag_field}"],
        "B_Eq": magnetic_field_variables[f"B_Eq_{mag_field}"],
        "L_m": magnetic_field_variables[f"L_m_{mag_field}"],
        "Alpha_Eq": magnetic_field_variables[f"Alpha_Eq_{mag_field}"],
        "Position": variables["xGEO"],
        "PSD": psd_var,
        "InvMu": magnetic_field_variables[f"InvMu_{mag_field}"],
        "InvK": magnetic_field_variables[f"InvK_{mag_field}"],
    }

    if calculate_Lstar:
        variables_to_save["L_star"] = magnetic_field_variables[f"L_star_{mag_field}"]

    saving_strategy = lanl_geogrid_sopa_esp_electron_strategy(processed_data_path, mag_field, satellite)

    ep.save(variables_to_save, saving_strategy, start_time, end_time, time_var=binned_time_var, append=append)

    if save_indices_sw:
        process_solar_wind_indices(
            start_time,
            end_time,
            processed_data_path=processed_data_path,
            bin_cadence=bin_cadence,
        )


PLOT_DEFAULTS = {
    "panels": ["Kp", "FEDU", "MLT"],
    "FEDU": {"pitch_angle": 90 * u.deg, "energies": [125, 600, 2000] * u.keV},
}

if __name__ == "__main__":
    ep.run_recipe_cli(
        process_lanl_geogrid_sopa_esp_electrons,
        strategy=lanl_geogrid_sopa_esp_electron_strategy,
        plot_defaults=PLOT_DEFAULTS,
    )
