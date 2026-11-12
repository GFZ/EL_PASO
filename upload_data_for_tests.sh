#!/bin/bash

# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

# Upload the system-test reference data to Hugging Face and pin the new revision in
# download_data_for_tests.sh.
# Requires write access to the spsw organisation: log in with `hf auth login` or set HF_TOKEN.
#
# Usage: bash upload_data_for_tests.sh ["commit message"]
#   uploads tests/system/data/, e.g. after `pytest ... --renew_solution true`
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ID="spsw/EL-PASO-Test-Data"
LOCAL_DIR="./tests/system/data"
REMOTE_DIR="data"
DOWNLOAD_SCRIPT="download_data_for_tests.sh"
commit_message="${1:-Update test data}"

if [ ! -d "${LOCAL_DIR}" ]; then
    echo "${LOCAL_DIR} does not exist, nothing to upload." >&2
    exit 1
fi

echo "Uploading ${LOCAL_DIR} to ${REPO_ID}/${REMOTE_DIR}..."
# --delete removes remote files that no longer exist locally
uvx --from huggingface_hub hf upload "${REPO_ID}" "${LOCAL_DIR}" "${REMOTE_DIR}" \
    --repo-type dataset --delete "${REMOTE_DIR}/*" --commit-message "${commit_message}"

revision=$(uvx --from huggingface_hub hf datasets info "${REPO_ID}" --format json \
    | python3 -c "import json, sys; print(json.load(sys.stdin)['sha'])")

sed -i "s/^REVISION=\"[^\"]*\"/REVISION=\"${revision}\"/" "${DOWNLOAD_SCRIPT}"
echo "Pinned ${REPO_ID}@${revision} in ${DOWNLOAD_SCRIPT}. Commit that change together with your code."
