# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: INP001

import cdflib
from tabulate import tabulate


def inspect_cdf_file(file_path: str) -> None:
    """Prints a formatted table of metadata for all variables in a CDF file.

    This function opens a CDF (Common Data Format) file, retrieves key metadata
    for each variable, and presents it in a clear, human-readable table. The table
    includes the variable name, data type, units, data shape, fill value, and a
    description. This is useful for quickly understanding the contents and
    structure of a CDF file.

    Parameters:
        file_path (str): The path to the CDF file.

    Raises:
        cdflib.CDFError: If the file is not a valid CDF file or an error occurs
                         while reading it.
    """
    cdf_file = cdflib.CDF(file_path)

    variable_names = cdf_file.cdf_info().zVariables

    var_attrs_to_print = []

    for var in variable_names:
        var_attrs_full = cdf_file.varattsget(var)
        vdr_info = cdf_file.varinq(var)
        if vdr_info.Last_Rec < 0:
            print(f"{var}: no records")
            continue
        var_data = cdf_file.varget(var)

        var_shape = var_data.shape  # type: ignore[reportAttributeAccessIssue]

        units = var_attrs_full.get("UNITS", "")

        desc = var_attrs_full.get("CATDESC", "")

        fillvall = var_attrs_full.get("FILLVAL", "")

        data_type = vdr_info.Data_Type_Description

        var_attrs_to_print.append([var, data_type, units, var_shape, fillvall, desc])

    print(  # noqa: T201
        tabulate(
            var_attrs_to_print,
            headers=["Variable name", "Data Type", "Units", "Data Shape", "Fill value", "Description"],
        )
    )


if __name__ == "__main__":
    file_name = "X"
    inspect_cdf_file(file_name)
