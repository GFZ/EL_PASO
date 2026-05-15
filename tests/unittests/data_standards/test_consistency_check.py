# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from el_paso.data_standard import ConsistencyCheck

TIME_LEN = 100
ENERGY_LEN = 20
PITCH_ANGLE_LEN = 50


@pytest.mark.basic
def test_consistency_correct():
    consistency_check = ConsistencyCheck()

    consistency_check.check_size(TIME_LEN, "Time", "call1")
    consistency_check.check_size(TIME_LEN, "Time", "call2")

    consistency_check.check_size(ENERGY_LEN, "Energy", "call1")
    consistency_check.check_size(ENERGY_LEN, "Energy", "call2")

    consistency_check.check_size(PITCH_ANGLE_LEN, "Pitch Angle", "call1")
    consistency_check.check_size(PITCH_ANGLE_LEN, "Pitch Angle", "call2")


@pytest.mark.basic
def test_consistency_time_wrong():
    consistency_check = ConsistencyCheck()

    consistency_check.check((TIME_LEN,), ["Time"], "call1")

    with pytest.raises(ValueError, match=r"Length mismatch! Time length of variable call1: *"):
        consistency_check.check((TIME_LEN + 1,), ["Time"], "call2")


@pytest.mark.basic
def test_consistency_energy_wrong():
    consistency_check = ConsistencyCheck()

    consistency_check.check((TIME_LEN, ENERGY_LEN, PITCH_ANGLE_LEN), ["Time", "Energy", "Alpha"], "call1")
    consistency_check.check((TIME_LEN, ENERGY_LEN, PITCH_ANGLE_LEN), ["Time", "Energy", "Alpha"], "call2")

    with pytest.raises(ValueError, match=r"Length mismatch! Energy length of variable call1: *"):
        consistency_check.check((TIME_LEN, ENERGY_LEN+1, PITCH_ANGLE_LEN), ["Time", "Energy", "Alpha"], "call2")

@pytest.mark.basic
def test_numbers():
    consistency_check = ConsistencyCheck()

    consistency_check.check((TIME_LEN, 2, PITCH_ANGLE_LEN), ["Time", 2, "Alpha"], "call1")
    consistency_check.check((TIME_LEN, 2, PITCH_ANGLE_LEN), ["Time", 2, "Alpha"], "call2")

    with pytest.raises(ValueError, match="Length mismatch! Variable call2 should have length 2, but encountered 3!"):
        consistency_check.check((TIME_LEN, 3, PITCH_ANGLE_LEN), ["Time", 2, "Alpha"], "call2")
