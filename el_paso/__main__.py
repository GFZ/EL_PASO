# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Allow the command line interface to be run as ``python -m el_paso``."""

from el_paso.cli.app import main

if __name__ == "__main__":
    main()
