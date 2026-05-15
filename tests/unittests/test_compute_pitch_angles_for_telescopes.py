# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from astropy import units as u

import el_paso as ep
from el_paso.processing import compute_pitch_angles_for_telescopes


def test_goes_pitch_angles_basic() -> None:

    b_brf = np.array([[15.788571, -89.67072 , 13.318663 ],
                      [15.338842, -88.74018 , 14.8691025],
                      [16.405478, -88.55671 , 13.420559 ],
                      [15.713195, -87.85792 , 14.339679 ],
                      [15.933415, -88.176674, 12.266859 ]], dtype=np.float32)

    true_pitch_angles = np.array([[116.13, 47.351, 150.02, 81.678, 15.157],
                                  [115.1,  46.284, 149.08, 80.625, 14.232],
                                  [115.92, 47.269, 149.71, 81.525, 15.358],
                                  [115.31, 46.574, 149.21, 80.873, 14.634],
                                  [116.62, 47.908, 150.43, 82.205, 15.728]], dtype=np.float32)

    bx = -b_brf[:,2]
    by =  b_brf[:,1]
    bz =  b_brf[:,0]

    b_rot = np.vstack([bx, by, bz]).T

    tele_alpha_angle = np.array([0., 0., 0., 0., 0.])
    tele_beta_angle = np.array([-35., 35., -70., 0, 70.])

    b_var = ep.Variable(data=b_rot, original_unit=u.nT)
    alpha_angles_var = ep.Variable(data=tele_alpha_angle, original_unit=u.deg)
    beta_angles_var = ep.Variable(data=tele_beta_angle, original_unit=u.deg)

    pitch_angles = compute_pitch_angles_for_telescopes(b_var, alpha_angles_var, beta_angles_var)

    pitch_angles_data = pitch_angles.get_data("deg")

    assert pitch_angles_data.shape == true_pitch_angles.shape
    assert pitch_angles_data == pytest.approx(true_pitch_angles, rel=1e-3)
