# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -e

DOI="10.5281/zenodo.20486611"
OUTPUT_DIR="./tests/system/"

echo "Downloading latest version of Zenodo DOI ${DOI} into ${OUTPUT_DIR}..."
uv run zenodo_get --doi "${DOI}" --output-dir "${OUTPUT_DIR}"

# Extract zip archive
unzip -o "${OUTPUT_DIR}/el-paso-test-data.zip" -d "${OUTPUT_DIR}"

# Cleanup
rm "${OUTPUT_DIR}/el-paso-test-data.zip"

echo "Done!"
