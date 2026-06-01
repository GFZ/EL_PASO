#!/bin/bash
set -e

RECORD_ID="20486612"
OUTPUT_DIR="./tests/system/"
API_URL="https://zenodo.org/api/records/${RECORD_ID}"

echo "Fetching metadata for Zenodo ID: ${RECORD_ID}..."
RAW_JSON=$(curl -s "${API_URL}")

# 1. Isolate the "files": [...] array block from the rest of the text
FILES_SECTION=$(echo "$RAW_JSON" | grep -o '"files": *\[.*\]')

# 2. Extract keys and download URLs exclusively from the files section
echo "$FILES_SECTION" | grep -o '"links": *{[^}]*}' | grep -o '"self": *"[^"]*"' | sed 's/"self": *"//;s/"//' > urls.txt
echo "$FILES_SECTION" | grep -o '"key": *"[^"]*"' | sed 's/"key": *"//;s/"//' > filenames.txt

echo "Starting downloads..."

# 3. Read links and names in parallel and fetch via curl
while read -u 3 URL && read -u 4 FILENAME; do
    if [ -n "$URL" ]; then
        echo "Downloading: ${FILENAME}"
        curl -L -C - -o "${OUTPUT_DIR}/${FILENAME}" "${URL}"
    fi
done 3<urls.txt 4<filenames.txt

# 4. Extract zip archive
unzip -o tests/system/el-paso-test-data.zip -d tests/system/

# Cleanup
rm urls.txt filenames.txt
rm tests/system/el-paso-test-data.zip

echo "Done!"