#!/usr/bin/env bash
# ==============================================================================
# Setup script for local UTA (PostgreSQL) and SeqRepo for offline HGVS throughput
# ==============================================================================
set -euo pipefail

SEQREPO_BASE_DIR="${HOME}/.local/share/seqrepo"
SEQREPO_INSTANCE="latest"
UTA_PORT="5432"

echo "=== 1. Starting local PostgreSQL UTA container ==="
docker compose -f docker-compose.uta.yml up -d

echo "Waiting for UTA PostgreSQL database to become ready..."
until docker exec uta_db pg_isready -U anonymous -d uta > /dev/null 2>&1; do
    echo "UTA container is starting up, waiting 5 seconds..."
    sleep 5
done
echo "UTA PostgreSQL container is ready!"

echo "=== 2. Setting up local SeqRepo sequence repository ==="
mkdir -p "${SEQREPO_BASE_DIR}"

if [ ! -d "${SEQREPO_BASE_DIR}/${SEQREPO_INSTANCE}" ]; then
    echo "Pulling SeqRepo data (this will download reference sequences locally)..."
    # Option A: Pull using docker
    docker run --rm \
        -v "${SEQREPO_BASE_DIR}:/usr/local/share/seqrepo" \
        biocommons/seqrepo:latest \
        seqrepo -r /usr/local/share/seqrepo pull
    
    # Or link instance if necessary
    if [ ! -d "${SEQREPO_BASE_DIR}/${SEQREPO_INSTANCE}" ] && [ -d "${SEQREPO_BASE_DIR}" ]; then
        LATEST_DIR=$(ls -dt "${SEQREPO_BASE_DIR}"/2* 2>/dev/null | head -n 1 || true)
        if [ -n "${LATEST_DIR}" ]; then
            ln -sfn "${LATEST_DIR}" "${SEQREPO_BASE_DIR}/latest"
        fi
    fi
else
    echo "SeqRepo directory already exists at ${SEQREPO_BASE_DIR}/${SEQREPO_INSTANCE}"
fi

echo "=== 3. Environment configuration ==="
export UTA_DB_URL="postgresql://anonymous@localhost:${UTA_PORT}/uta"
export HGVS_SEQREPO_DIR="${SEQREPO_BASE_DIR}/${SEQREPO_INSTANCE}"

echo "Configured environment variables:"
echo "  UTA_DB_URL=${UTA_DB_URL}"
echo "  HGVS_SEQREPO_DIR=${HGVS_SEQREPO_DIR}"
echo ""
echo "To export these in your current shell or .bashrc:"
echo "  export UTA_DB_URL=\"${UTA_DB_URL}\""
echo "  export HGVS_SEQREPO_DIR=\"${HGVS_SEQREPO_DIR}\""
echo ""
echo "You can now run:"
echo "  python process_funk.py --input urn_mavedb_00001277-a-1_custom.csv --output urn_mavedb_00001277-a-1_spdi.csv"

