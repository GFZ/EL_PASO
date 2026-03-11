# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import os
import typing
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd
import requests
import scipy as sp  # type: ignore[reportMissingTypeStubs]
import swvo.io as swvo_io  # type: ignore[reportMissingTypeStubs]
from astropy import units as u  # type: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

import el_paso as ep
from el_paso.utils import enforce_utc_timezone

SW_Index = Literal["Kp", "SW_speed", "SW_density", "Dst", "Pdyn", "IMF_Bz", "IMF_By", "G1", "G2", "G3", "W_params"]


@overload
def load_indices_solar_wind_parameters(
    start_time: datetime,
    end_time: datetime,
    requested_outputs: Iterable[SW_Index],
    target_time_variable: None = None,
    *,
    w_parameter_method: Literal["TsyWebsite", "Calculation"] = "Calculation",
) -> dict[SW_Index, tuple[ep.Variable, ep.Variable]]: ...


@overload
def load_indices_solar_wind_parameters(
    start_time: datetime,
    end_time: datetime,
    requested_outputs: Iterable[SW_Index],
    target_time_variable: ep.Variable,
    *,
    w_parameter_method: Literal["TsyWebsite", "Calculation"] = "Calculation",
) -> dict[SW_Index, ep.Variable]: ...


def load_indices_solar_wind_parameters(  # noqa: C901, PLR0912, PLR0915
    start_time: datetime,
    end_time: datetime,
    requested_outputs: Iterable[SW_Index],
    target_time_variable: ep.Variable | None = None,
    *,
    w_parameter_method: Literal["TsyWebsite", "Calculation"] = "Calculation",
) -> dict[SW_Index, tuple[ep.Variable, ep.Variable]] | dict[SW_Index, ep.Variable]:
    """Loads a variety of space weather indices and solar wind parameters.

    This function fetches and processes data for several common space weather
    and solar wind indices, including Kp, Dst, solar wind plasma properties,
    and Tsyganenko model parameters (G1, G2, G3, W_params).

    Data is downloaded and cached locally to a `.elpaso` directory in the user's home
    directory. The function can either return the data with its original timestamps
    or interpolate the data to a new set of timestamps provided by a `target_time_variable`.

    Parameters:
        start_time (datetime): The start time for the data retrieval.
        end_time (datetime): The end time for the data retrieval.
        requested_outputs (Iterable[SW_Index]): A list of space weather indices to load.
                                                Supported values are defined by the `SW_Index` Literal.
        target_time_variable (ep.Variable | None): An optional `ep.Variable` containing the
                                                    target timestamps for interpolation. If `None`,
                                                    the raw data and its timestamps are returned.
        w_parameter_method ('TsyWebsite' | 'Calculation'): The method to use for obtaining
                                                            W_params. 'TsyWebsite' fetches data from
                                                            the Tsyganenko website (only available until 2023),
                                                            while 'Calculation' computes them from other
                                                            solar wind data. Defaults to 'Calculation'.

    Returns:
        dict[SW_Index, ep.Variable | tuple[ep.Variable, ep.Variable]]: A dictionary where each key
                                                                        is a requested index and the value
                                                                        is the corresponding `ep.Variable`
                                                                        object(s). If `target_time_variable`
                                                                        is provided, the value is a single
                                                                        `ep.Variable`. Otherwise, it is a
                                                                        tuple of `(data_variable, time_variable)`.

    Raises:
        TypeError: If `requested_outputs` is not an iterable of strings.
        OSError: If the HOME environment variable is not set.
        ValueError: If a requested output is not a supported `SW_Index` or if
                    an unsupported method is requested.
        FileNotFoundError: If the data file from the Tsyganenko website is not found.
    """
    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    if not isinstance(requested_outputs, list):
        msg = "requested_outputs must be a list of strings!"
        raise TypeError(msg)

    result_dict: dict[SW_Index, tuple[ep.Variable, ep.Variable]] | dict[SW_Index, ep.Variable] = {}

    home_path = os.getenv("HOME")
    if home_path is None:
        msg = "HOME environment variable is not set!"
        raise OSError(msg)

    base_data_path = Path(home_path) / ".elpaso"

    for requested_output in requested_outputs:
        match requested_output:
            case "Kp":
                kp_model_order: list[swvo_io.kp.KpModel] = [
                    swvo_io.kp.KpOMNI(base_data_path / "OMNI_low_res"),
                    swvo_io.kp.KpNiemegk(base_data_path / "KpNiemegk"),
                ]
                output_df = swvo_io.kp.read_kp_from_multiple_models(
                    start_time, end_time, model_order=kp_model_order, download=True
                )

                assert isinstance(output_df, pd.DataFrame)

                result = _create_variables_from_data_frame(
                    output_df, "kp", u.dimensionless_unscaled, target_time_variable, "previous"
                )

            case "Dst":
                output_df = swvo_io.dst.DSTOMNI(base_data_path / "OMNI_low_res").read(
                    start_time, end_time, download=True
                )

                result = _create_variables_from_data_frame(output_df, "dst", u.nT, target_time_variable, "linear")

            case "Pdyn":
                output_df = _cache_omni_high_res(base_data_path, start_time, end_time)
                assert isinstance(output_df, pd.DataFrame)

                output_df["pdyn"] = output_df["pdyn"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "pdyn", u.nPa, target_time_variable, "linear")

            case "IMF_Bz":
                # we request two additional hours for interpolation
                output_df = _cache_omni_high_res(base_data_path, start_time, end_time)
                assert isinstance(output_df, pd.DataFrame)

                output_df["bz_gsm"] = output_df["bz_gsm"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "bz_gsm", u.nT, target_time_variable, "linear")

            case "IMF_By":
                # we request two additional hours for interpolation
                output_df = _cache_omni_high_res(base_data_path, start_time, end_time)
                assert isinstance(output_df, pd.DataFrame)

                output_df["by_gsm"] = output_df["by_gsm"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "by_gsm", u.nT, target_time_variable, "linear")

            case "SW_speed":
                # we request two additional hours for interpolation
                output_df = _cache_omni_high_res(base_data_path, start_time, end_time)
                assert isinstance(output_df, pd.DataFrame)

                output_df["speed"] = output_df["speed"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(
                    output_df,
                    "speed",
                    u.km * u.s**-1,
                    target_time_variable,
                    "linear",  # type: ignore[reportUnknownArgumentType]
                )

            case "SW_density":
                # we request two additional hours for interpolation
                output_df = _cache_omni_high_res(base_data_path, start_time, end_time)
                assert isinstance(output_df, pd.DataFrame)

                output_df["proton_density"] = (
                    output_df["proton_density"].interpolate(method="spline", order=3).ffill().bfill()
                )
                output_df["proton_density"] = output_df["proton_density"].clip(lower=0)

                result = _create_variables_from_data_frame(
                    output_df, "proton_density", u.cm**-3, target_time_variable, "linear"
                )

            case "G1":
                g1_var, time_var = _calculate_g1(start_time, end_time, target_time_variable)
                result = (g1_var, time_var) if target_time_variable is None else g1_var

            case "G2":
                g2_var, time_var = _calculate_g2(start_time, end_time, target_time_variable)
                result = (g2_var, time_var) if target_time_variable is None else g2_var

            case "G3":
                g3_var, time_var = _calculate_g3(start_time, end_time, target_time_variable)
                result = (g3_var, time_var) if target_time_variable is None else g3_var

            case "W_params":
                w_var, time_var = _get_w_parameters(start_time, end_time, target_time_variable, w_parameter_method)

                result = (w_var, time_var) if target_time_variable is None else w_var

            case _:
                msg = f"Requested invalid output: {requested_output}!"
                raise ValueError(msg)

        result_dict[requested_output] = result  # type: ignore[reportArgumentType]

    return result_dict


def _create_variables_from_data_frame(
    df_in: pd.DataFrame,
    data_key: str,
    unit: u.UnitBase,
    target_time_variable: ep.Variable | None,
    time_interp_method: str,
) -> ep.Variable | tuple[ep.Variable, ep.Variable]:
    data_var = ep.Variable(data=df_in[data_key].to_numpy(), original_unit=unit)
    timestamps = np.asarray([t.timestamp() for t in df_in.index.to_pydatetime()])  # type: ignore[reportUnknownMemberType]
    time_var = ep.Variable(data=timestamps, original_unit=ep.units.posixtime)  # type: ignore[reportUnknownArgumentType]

    if target_time_variable is None:
        result = (data_var, time_var)
    else:
        f = sp.interpolate.interp1d(time_var.get_data(), data_var.get_data(), kind=time_interp_method)
        data_var.set_data(f(target_time_variable.get_data(ep.units.posixtime)), "same")
        result = data_var

    return result


@lru_cache
def _cache_omni_high_res(base_data_path: Path, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    output_df = swvo_io.solar_wind.SWOMNI(base_data_path / "OMNI_high_res").read(
        start_time - timedelta(hours=1),
        end_time + timedelta(hours=1),
        download=True,
    )

    return output_df


def _calculate_g1(
    start_time: datetime, end_time: datetime, target_time_variable: ep.Variable | None
) -> tuple[ep.Variable, ep.Variable]:
    additional_required_inputs = typing.cast("list[SW_Index]", ["SW_speed", "IMF_Bz", "IMF_By"])

    inputs = load_indices_solar_wind_parameters(start_time, end_time, additional_required_inputs, None)

    sw_speed = inputs["SW_speed"][0].get_data().astype(np.float64)
    sw_speed_time = inputs["SW_speed"][1].get_data(ep.units.posixtime).astype(np.float64)

    imf_bz = inputs["IMF_Bz"][0].get_data().astype(np.float64)
    imf_bz_time = inputs["IMF_Bz"][1].get_data(ep.units.posixtime).astype(np.float64)

    imf_by = inputs["IMF_By"][0].get_data().astype(np.float64)
    imf_by_time = inputs["IMF_By"][1].get_data(ep.units.posixtime).astype(np.float64)

    if not np.array_equal(sw_speed_time, imf_bz_time) or not np.array_equal(sw_speed_time, imf_by_time):
        msg = "Time variables of SW_speed, IMF_Bz, and IMF_By must be equal!"
        raise ValueError(msg)

    b_perp = np.sqrt(imf_bz**2 + imf_by**2)
    theta = np.atan2(imf_by, imf_bz)

    timestamps = sw_speed_time
    g1: list[float] = []

    for it, curr_time in enumerate(timestamps):
        idx = np.argwhere(np.abs(timestamps[: it + 1] - curr_time) <= timedelta(hours=1).total_seconds())

        g1.append(
            float(
                np.nanmean(
                    sw_speed[idx] * (b_perp[idx] / 40) ** 2 / (1 + b_perp[idx] / 40) * np.sin(theta[idx] / 2) ** 3
                )
            )
        )

    data_var = ep.Variable(data=np.asarray(g1), original_unit=u.dimensionless_unscaled)
    time_var = ep.Variable(data=timestamps, original_unit=ep.units.posixtime)

    if target_time_variable is not None:
        f = sp.interpolate.interp1d(time_var.get_data(), data_var.get_data(), kind="linear")
        data_var.set_data(f(target_time_variable.get_data(ep.units.posixtime)), "same")

    return data_var, time_var


def _calculate_g2(
    start_time: datetime, end_time: datetime, target_time_variable: ep.Variable | None
) -> tuple[ep.Variable, ep.Variable]:
    additional_required_inputs = typing.cast("list[SW_Index]", ["SW_speed", "IMF_Bz"])

    inputs = load_indices_solar_wind_parameters(start_time, end_time, additional_required_inputs, None)

    sw_speed = inputs["SW_speed"][0].get_data().astype(np.float64)
    sw_speed_time = inputs["SW_speed"][1].get_data(ep.units.posixtime).astype(np.float64)

    imf_bz = inputs["IMF_Bz"][0].get_data().astype(np.float64)
    imf_bz_time = inputs["IMF_Bz"][1].get_data(ep.units.posixtime).astype(np.float64)

    if not np.array_equal(sw_speed_time, imf_bz_time):
        msg = "Time variables of SW_speed, and IMF_Bz must be equal!"
        raise ValueError(msg)

    b_south = np.where(imf_bz < 0, -imf_bz, 0)

    timestamps = sw_speed_time
    g2: list[float] = []

    for it, curr_time in enumerate(timestamps):
        idx = np.argwhere(np.abs(timestamps[: it + 1] - curr_time) <= timedelta(hours=1).total_seconds())

        g2.append(float(np.nanmean(sw_speed[idx] * b_south[idx] / 200)))

    data_var = ep.Variable(data=np.asarray(g2), original_unit=u.dimensionless_unscaled)
    time_var = ep.Variable(data=timestamps, original_unit=ep.units.posixtime)

    if target_time_variable is not None:
        f = sp.interpolate.interp1d(time_var.get_data(), data_var.get_data(), kind="linear")
        data_var.set_data(f(target_time_variable.get_data(ep.units.posixtime)), "same")

    return data_var, time_var


def _calculate_g3(
    start_time: datetime, end_time: datetime, target_time_variable: ep.Variable | None
) -> tuple[ep.Variable, ep.Variable]:
    additional_required_inputs = typing.cast("list[SW_Index]", ["SW_speed", "SW_density", "IMF_Bz"])

    inputs = load_indices_solar_wind_parameters(start_time, end_time, additional_required_inputs, None)

    sw_speed = inputs["SW_speed"][0].get_data().astype(np.float64)
    sw_speed_time = inputs["SW_speed"][1].get_data(ep.units.posixtime).astype(np.float64)

    sw_density = inputs["SW_density"][0].get_data().astype(np.float64)
    sw_density_time = inputs["SW_density"][1].get_data(ep.units.posixtime).astype(np.float64)

    imf_bz = inputs["IMF_Bz"][0].get_data().astype(np.float64)
    imf_bz_time = inputs["IMF_Bz"][1].get_data(ep.units.posixtime).astype(np.float64)

    if not np.array_equal(sw_speed_time, imf_bz_time) or not np.array_equal(sw_speed_time, sw_density_time):
        msg = "Time variables of SW_speed, SW_density, and IMF_Bz must be equal!"
        raise ValueError(msg)

    b_south = np.where(imf_bz < 0, -imf_bz, 0)

    timestamps = sw_speed_time
    g3: list[float] = []

    for it, curr_time in enumerate(timestamps):
        idx = np.argwhere(np.abs(timestamps[: it + 1] - curr_time) <= timedelta(hours=1).total_seconds())

        g3.append(float(np.nanmean(sw_density[idx] * sw_speed[idx] * b_south[idx] / 2000)))

    data_var = ep.Variable(data=np.asarray(g3), original_unit=u.dimensionless_unscaled)
    time_var = ep.Variable(data=timestamps, original_unit=ep.units.posixtime)

    if target_time_variable is not None:
        f = sp.interpolate.interp1d(time_var.get_data(), data_var.get_data(), kind="linear")
        data_var.set_data(f(target_time_variable.get_data(ep.units.posixtime)), "same")

    return data_var, time_var


def _get_w_parameters(
    start_time: datetime,
    end_time: datetime,
    target_time_variable: ep.Variable | None,
    w_parameter_method: Literal["TsyWebsite", "Calculation"],
) -> tuple[ep.Variable, ep.Variable]:
    match w_parameter_method:
        case "TsyWebsite":
            if target_time_variable is None:
                msg = "W parameters from Tsyganenko's website is only available if target_time_variable is set!"
                raise ValueError(msg)

            w_params = _get_w_parameters_tsyganenko(target_time_variable)

            w_data = np.empty((len(w_params["W1"]), 6))
            w_data[:, 0] = w_params["W1"]
            w_data[:, 1] = w_params["W2"]
            w_data[:, 2] = w_params["W3"]
            w_data[:, 3] = w_params["W4"]
            w_data[:, 4] = w_params["W5"]
            w_data[:, 5] = w_params["W6"]

            w_var = ep.Variable(data=w_data, original_unit=u.dimensionless_unscaled)

            return (w_var, target_time_variable)

        case "Calculation":
            return _calculate_w_parameters(start_time, end_time, target_time_variable)


# Tsyganenko, N. A. & Sitnov, M. I. Modeling the dynamics of the inner magnetosphere during strong geomagnetic storms.
# Journal of Geophysical Research: Space Physics 110, (2005).

# These parameters are the new ones from Tsyganenko's website; the ones in the
# original paper are slightly different
LAMBDA = [0.394732, 0.550920, 0.387365, 0.436819, 0.405553, 1.26131]
BETA = [0.846509, 0.180725, 2.26596, 1.28211, 1.62290, 2.42297]
GAMMA = [0.916555, 0.898772, 1.29123, 1.33199, 0.699074, 0.537116]
R = [0.383403, 0.648176, 0.318752e-01, 0.581168, 1.15070, 0.843004]


def _calculate_w_parameters(
    start_time: datetime, end_time: datetime, target_time_variable: ep.Variable | None
) -> tuple[ep.Variable, ep.Variable]:
    additional_required_inputs = typing.cast("list[SW_Index]", ["IMF_Bz", "SW_speed", "SW_density"])

    # shift by two weeks because the values are influenced by past conditions
    start_time_sifted = start_time - timedelta(days=14)

    start_timestamp = start_time_sifted.timestamp()
    end_timestamp = end_time.timestamp()
    cadence_minutes = 5
    timestamps = np.arange(start_timestamp, end_timestamp + 10, cadence_minutes * 60)
    timestamps_minutes = timestamps / 60

    # calculation requires 5 min resolution
    time_var_calculation = ep.Variable(data=timestamps, original_unit=ep.units.posixtime)

    inputs = load_indices_solar_wind_parameters(
        start_time_sifted, end_time, additional_required_inputs, time_var_calculation
    )

    sw_speed = inputs["SW_speed"].get_data().astype(np.float64)
    sw_density = inputs["SW_density"].get_data().astype(np.float64)
    imf_bz = inputs["IMF_Bz"].get_data().astype(np.float64)

    b_south = np.where(imf_bz < 0, -imf_bz, 0)

    w_params = np.full((len(sw_speed), 6), 0.0)

    cutoff_value = -10  # same as in Tsyganenko's code
    sw_density_with_He = sw_density * 1.16  # same as in Tsyganenko's code  # noqa: N806
    sw_density_normed = sw_density_with_He / 5.0
    sw_speed_normed = sw_speed / 400.0
    b_south_normed = b_south / 5.0

    for idx in range(len(w_params)):
        for i in range(6):
            decay = -R[i] / 60 * (timestamps_minutes[idx] - timestamps_minutes)
            mask_to_sum = (decay > cutoff_value) & (decay < 0)

            s_tmp = (
                sw_density_normed[mask_to_sum] ** LAMBDA[i]
                * sw_speed_normed[mask_to_sum] ** BETA[i]
                * b_south_normed[mask_to_sum] ** GAMMA[i]
            )

            w_params[idx, i] = (
                R[i]
                / 60
                * cadence_minutes
                * np.sum(s_tmp * np.exp(-R[i] / 60 * (timestamps_minutes[idx] - timestamps_minutes[mask_to_sum])))
            )

    if target_time_variable is not None:
        w_params_interp = np.full((len(target_time_variable.get_data()), 6), np.nan)
        for i in range(6):
            f = sp.interpolate.interp1d(timestamps, w_params[:, i], kind="linear", fill_value="extrapolate")
            w_params_interp[:, i] = f(target_time_variable.get_data(ep.units.posixtime))

        w_data_to_return = w_params_interp
        time_var_to_return = target_time_variable
    else:
        w_data_to_return = w_params
        time_var_to_return = time_var_calculation

    w_var = ep.Variable(data=w_data_to_return, original_unit=u.dimensionless_unscaled)
    w_var.truncate(time_var_to_return, start_time, end_time)
    time_var_to_return.truncate(time_var_to_return, start_time, end_time)

    return (w_var, time_var_to_return)


MAX_YEAR_AVAILABLE = 2023


def _get_w_parameters_tsyganenko(target_time_variable: ep.Variable) -> dict[str, NDArray[np.float64]]:
    timestamps = target_time_variable.get_data(ep.units.posixtime).astype(np.float64)

    datetimes = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    years = np.unique([dt.year for dt in datetimes])

    if years[-1] > MAX_YEAR_AVAILABLE:
        msg = "W parameters from Tsyganenko's website are only available until 2023!"
        raise ValueError(msg)

    w_params: dict[str, list[float]] = {"W1": [], "W2": [], "W3": [], "W4": [], "W5": [], "W6": []}

    for year in years:
        url = f"https://geo.phys.spbu.ru/~tsyganenko/models/ts05/{year:d}_OMNI_5m_with_TS05_variables.dat"

        response = requests.get(url, stream=True, verify=False, timeout=10)  # noqa: S501

        response.raise_for_status()

        df = pd.read_csv(  # type: ignore[reportUnknownMemberType]
            StringIO(response.text),
            names=["Year", "Day", "Hour", "Min", "W1", "W2", "W3", "W4", "W5", "W6"],
            usecols=(0, 1, 2, 3, 17, 18, 19, 20, 21, 22),  # type: ignore[reportUnknownMemberType]
            sep=r"\s+",
        )

        timestamps_data: list[float] = []

        for _, row in df.loc[:, ["Year", "Day", "Hour", "Min"]].iterrows():
            year_ = int(row["Year"])
            day = int(row["Day"])
            hour = int(row["Hour"])
            minute = int(row["Min"])

            dt = datetime.strptime(f"{year_:04d}-{day:03d}-{hour:02d}-{minute:02d}", "%Y-%j-%H-%M").replace(
                tzinfo=timezone.utc
            )
            timestamps_data.append(dt.replace(tzinfo=timezone.utc).timestamp())

        # find the timestamps for the current year
        timestamp_year_begin = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
        timestamp_year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()

        curr_year_idx = (timestamps >= timestamp_year_begin) & (timestamps <= timestamp_year_end)

        for w_str in ["W1", "W2", "W3", "W4", "W5", "W6"]:
            w_param = df[w_str].to_numpy()
            w_data = np.interp(timestamps[curr_year_idx], timestamps_data, w_param)

            w_params[w_str] += list(w_data)

    return {key: np.asarray(data).astype(np.float64) for key, data in w_params.items()}
