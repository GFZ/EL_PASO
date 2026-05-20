# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Optional  # noqa: UP035

import numpy as np
from swvo.io.RBMDataSet import RBMDataSet

from el_paso.dataset.interp_functions import TargetType

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.dataset import DataSet
    from el_paso.typing import MagneticFieldLiteral


def create_RBSP_line_data(
    start_time: datetime,
    end_time: datetime,
    data_server_path: Path,
    target_en: float | Iterable[float],
    target_al: float | Iterable[float],
    target_type: TargetType | Literal["TargetPairs", "TargetMeshGrid"],
    energy_offset_threshold: float = 0.1,
    instruments: Optional[list[str]] = None,
    satellites: Optional[list[str]] = None,
    mfm: MagneticFieldLiteral = "T89",
    *,
    adjust_targets: bool = True,
    verbose: bool = True,
) -> tuple[list[DataSet], list[str]]:
    """Create RBSP line data for specified energy and pitch angle targets.

    Loads and processes RBSP particle data for the requested time interval and
    extracts line data corresponding to the specified target energies and local
    pitch angles.

    Args:
        start_time (datetime): Start time of the data interval.
        end_time (datetime): End time of the data interval.
        data_server_path (Path): Path to the data server containing the RBSP datasets.
        target_en (float | Iterable[float]): Target energy or iterable of target energies in MeV.
        target_al (float | Iterable[float]): Target local pitch angle or iterable of local pitch angles in degrees.
        target_type (TargetType | Literal["TargetPairs", "TargetMeshGrid"]): Strategy used to combine energy and pitch
                        angle targets.
        energy_offset_threshold (float, optional): Maximum allowed relative energy offset between requested and \
                        available energies. Defaults to ``0.1``.
        instruments (Optional[list[str]], optional): Instruments to include in the processing. If ``None``,
                        defaults to ``["HOPE", "MAGEIS", "REPT"]``.
        satellites (Optional[list[str]], optional): RBSP satellites to use.
                        If ``None``, defaults to ``["RBSPA", "RBSPB"]``.
        mfm (MagneticFieldLiteral, optional): Magnetic field model used for calculations. Defaults to ``"T89"``.
        adjust_targets (bool, optional): If ``True``, targets are adjusted to the closest available values.
                        If ``False``, values are interpolated. Defaults to ``True``.
        verbose (bool, optional): If ``True``, print progress and diagnostic information during processing.
                        Defaults to ``True``.

    Returns:
        tuple[list[DataSet], list[str]]: Tuple containing the processed datasets and the list of instruments used.

    Raises:
        ValueError: If the provided target configuration is invalid.
        FileNotFoundError: If required RBSP data files cannot be found.
        RuntimeError: If no valid datasets could be created for the requested interval.
    """
    # Instruments represents also the priority of the instrument for overlapping energies. The first instrument will be prefered.  # noqa: E501

    instruments = instruments or ["HOPE", "MAGEIS", "REPT"]
    satellites = satellites or ["RBSPA", "RBSPB"]

    # pass and check args
    if isinstance(data_server_path, str):
        data_server_path = Path(data_server_path)
    if not isinstance(target_al, Iterable):
        target_al = [target_al]
    if not isinstance(target_en, Iterable):
        target_en = [target_en]
    if not isinstance(satellites, Iterable) or isinstance(satellites, str):
        satellites = [satellites]
    if isinstance(target_type, str):
        target_type = TargetType[target_type]

    if target_type == TargetType.TargetPairs:
        assert len(target_en) == len(target_al), "For TargetType.Pairs, the target vectors must have the same size!"  # ty:ignore[invalid-argument-type]

    result_arr = []
    list_instruments_used = []

    for satellite in satellites:
        rbm_data: list[DataSet] = []

        for i, instrument in enumerate(instruments):
            rbm_data.append(
                RBMDataSet(
                    satellite,  # ty: ignore[invalid-argument-type]
                    instrument,
                    mfm,
                    start_time,
                    end_time,
                    data_server_path,
                    verbose=verbose,
                )
            )

            # strip of time dimention
            if rbm_data[i].energy_channels.shape[0] == len(rbm_data[i].time):
                rbm_data[i].energy_channels_no_time = np.nanmean(rbm_data[i].energy_channels, axis=0)  # ty:ignore[unresolved-attribute]
            else:
                rbm_data[i].energy_channels_no_time = rbm_data[i].energy_channels  # ty:ignore[unresolved-attribute]
            if rbm_data[i].alpha_local.shape[0] == len(rbm_data[i].time):
                rbm_data[i].alpha_local_no_time = np.nanmean(rbm_data[i].alpha_local, axis=0)  # ty:ignore[unresolved-attribute]
            else:
                rbm_data[i].alpha_local_no_time = rbm_data[i].alpha_local  # ty:ignore[unresolved-attribute]

        for e, target_en_single in enumerate(target_en):
            if verbose:
                pass

            energy_offsets = np.empty((len(instruments),))

            for i, _instrument in enumerate(instruments):
                energy_offsets[i] = np.nanmin(
                    np.abs(rbm_data[i].energy_channels_no_time - target_en_single),
                    axis=None,
                )

                if verbose:
                    pass

                # initiate the RBMDataSet for the result
                if e == 0 and i == 0:
                    rbm_data_set_result = deepcopy(rbm_data[i])

                    if target_type == TargetType.TargetPairs:
                        rbm_data_set_result.line_data_flux = np.empty((len(rbm_data_set_result.time), len(target_en)))  # ty:ignore[invalid-argument-type, unresolved-attribute]
                        rbm_data_set_result.line_data_energy = np.empty((len(target_en),))  # ty:ignore[invalid-argument-type, unresolved-attribute]
                        rbm_data_set_result.line_data_alpha_local = np.empty((len(target_al),))  # ty:ignore[invalid-argument-type, unresolved-attribute]
                    elif target_type == TargetType.TargetMeshGrid:
                        rbm_data_set_result.line_data_flux = np.empty(  # ty:ignore[unresolved-attribute]
                            (
                                len(rbm_data_set_result.time),
                                len(target_en),  # ty:ignore[invalid-argument-type]
                                len(target_al),  # ty:ignore[invalid-argument-type]
                            )
                        )
                        rbm_data_set_result.line_data_energy = np.empty((len(target_en),))  # ty:ignore[invalid-argument-type, unresolved-attribute]
                        rbm_data_set_result.line_data_alpha_local = np.empty((len(target_al),))  # ty:ignore[invalid-argument-type, unresolved-attribute]

            energy_offsets_relative = energy_offsets / target_en_single
            if np.all(np.abs(energy_offsets_relative) > energy_offset_threshold):
                msg = f"For the given energy target ({target_en_single:.2e} MeV), no suitable energy channel"
                "was found for a threshold of {energy_offset_threshold:.02f}!"
                raise ValueError(msg)

            min_offset_instrument = np.argmax(np.abs(energy_offsets_relative) <= energy_offset_threshold)
            list_instruments_used.append(instruments[min_offset_instrument])

            if verbose:
                pass

            closest_en_idx = np.nanargmin(
                np.abs(rbm_data[min_offset_instrument].energy_channels_no_time - target_en_single)
            )
            rbm_data_set_result.line_data_energy[e] = rbm_data[min_offset_instrument].energy_channels_no_time[
                closest_en_idx
            ]

            if target_type == TargetType.TargetPairs:
                closest_al_idx = np.nanargmin(
                    np.abs(rbm_data[min_offset_instrument].alpha_local_no_time - target_al[e])  # ty:ignore[not-subscriptable]
                )
                rbm_data_set_result.line_data_alpha_local[e] = rbm_data[min_offset_instrument].alpha_local_no_time[
                    closest_al_idx
                ]

                if adjust_targets:
                    rbm_data_set_result.line_data_flux[:, e] = rbm_data[min_offset_instrument].Flux[
                        :, closest_en_idx, closest_al_idx
                    ]
                else:
                    rbm_data_set_result.line_data_flux[:, e] = np.squeeze(
                        rbm_data[min_offset_instrument].interp_flux(
                            target_en_single,
                            target_al[e],  # ty:ignore[not-subscriptable]
                            TargetType.TargetPairs,
                        )
                    )

            elif target_type == TargetType.TargetMeshGrid:
                for a, target_al_single in enumerate(target_al):
                    closest_al_idx = np.nanargmin(
                        np.abs(rbm_data[min_offset_instrument].alpha_local_no_time - target_al_single)
                    )
                    rbm_data_set_result.line_data_alpha_local[a] = rbm_data[min_offset_instrument].alpha_local_no_time[
                        closest_al_idx
                    ]

                    if adjust_targets:
                        rbm_data_set_result.line_data_flux[:, e, a] = rbm_data[min_offset_instrument].Flux[
                            :, closest_en_idx, closest_al_idx
                        ]
                    else:
                        rbm_data_set_result.line_data_flux[:, e, a] = np.squeeze(
                            rbm_data[min_offset_instrument].interp_flux(
                                target_en_single,
                                target_al_single,
                                TargetType.TargetPairs,
                            )
                        )

        result_arr.append(rbm_data_set_result)

    return result_arr, list_instruments_used
