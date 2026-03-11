# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

ParticleLiteral = Literal["electron", "proton", "helium", "oxygen"]


def rest_energy(species: ParticleLiteral) -> float:
    """Return the rest energy for the input species.

    Args:
        species (str): The species of particle ('electron', 'proton', 'helium', 'oxygen').

    Returns:
        np.ndarray or float: The rest energy of the species in MeV.

    Raises:
        ValueError: If an unknown species is provided.
    """
    # Rest energy in MeV for different species
    rest_energies = {
        "electron": 0.511,  # MeV
        "proton": 938.272,  # MeV
        "helium": 3727.379,  # MeV (Helium-4 nucleus)
        "oxygen": 14958.9,  # MeV (Oxygen-16 nucleus)
    }

    # Set mc2 based on the species
    if species.lower() in rest_energies:
        mc2 = rest_energies[species.lower()]
    else:
        msg = f"Unknown species '{species}'. Valid options are 'electron', 'proton', 'helium', 'oxygen'."
        raise ValueError(msg)

    return mc2


@overload
def en2pc(energy: float, species: ParticleLiteral = "electron") -> float: ...


@overload
def en2pc(energy: NDArray[np.number], species: ParticleLiteral = "electron") -> NDArray[np.number]: ...


def en2pc(energy: float | NDArray[np.number], species: ParticleLiteral = "electron") -> float | NDArray[np.number]:
    r"""Calculate the relativistic momentum (p*c) for a given total energy.

    This function calculates the relativistic energy using the formula:
    $$
    pc = \sqrt{(E/m_0c^2 + 1)^2 - 1} \cdot m_0c^2
    $$
    where $E$ is the total energy, $m_0c^2$ is the rest energy of the particle,
    and $pc$ is the relativistic momentum.

    Args:
        energy (np.ndarray or float): The total energy in MeV.
        species (str): The species of particle ('electron', 'proton', 'helium', 'oxygen').

    Returns:
        np.ndarray or float: The calculated relativistic momentum times c (p*c).

    """
    mc2 = rest_energy(species)
    # Calculate the relativistic energy

    return np.sqrt((energy / mc2 + 1) ** 2 - 1) * mc2
