# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import lazy_loader as lazy

ThemisProbe = Literal["a", "b", "c", "d", "e"]

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submod_attrs={
        "process_themis_fft_waves": ["process_themis_fft_waves", "themis_fft_waves_strategy"],
        "process_themis_scpot_density": ["process_themis_scpot_density", "themis_scpot_density_strategy"],
    },
)

__all__ = ["ThemisProbe", *_lazy_all]  # noqa: PLE0604  (lazy_loader supplies the names)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from el_paso.recipes.themis.process_themis_fft_waves import process_themis_fft_waves, themis_fft_waves_strategy
    from el_paso.recipes.themis.process_themis_scpot_density import (
        process_themis_scpot_density,
        themis_scpot_density_strategy,
    )

    __all__ = [
        "ThemisProbe",
        "process_themis_fft_waves",
        "process_themis_scpot_density",
        "themis_fft_waves_strategy",
        "themis_scpot_density_strategy",
    ]
