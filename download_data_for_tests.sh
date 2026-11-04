# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set -e

REPO_ID="spsw/EL-PASO-Test-Data"
REVISION="2796e90dc0707110ba0397b0891fb1575fe4581b"   # bump when the data changes
OUTPUT_DIR="./tests/system/"

echo "Downloading ${REPO_ID}@${REVISION}..."
uvx --from huggingface_hub hf download "${REPO_ID}" \
    --repo-type dataset --revision "${REVISION}" \
    --include "data/*" --local-dir "${OUTPUT_DIR}"
echo "Done!"
