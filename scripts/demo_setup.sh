#!/usr/bin/env bash
# One command from a fresh GitHub Codespace to a demo that is ready to present:
#
#   bash scripts/demo_setup.sh
#
# 1. MVTec AD bottle, wood and pill (official links, about 0.9 GB) are
#    downloaded into $MVTEC_DIR, outside the repository - only if missing.
# 2. The Docker Compose stack is built and started.
# 3. database/load_data.py registers the images in PostgreSQL - only while
#    the images table is empty, so a rerun never resets a prepared demo -
#    and the API container is recreated to see the new data/ folder.
# 4. Containers must reach each other and the API must answer /health/db.
# 5. scripts/demo_train.py trains the models in the Airflow order - only what
#    is missing for the newest released batch (first run about 55 minutes,
#    later runs: nothing).
# 6. scripts/demo_check.py judges the test images shown on the Prediction tab
#    and leaves enough recent predictions for the Evidently drift check;
#    both Airflow DAGs are unpaused and the Airflow login is printed.
#
# Dataset licence: MVTec AD, CC BY-NC-SA 4.0 (non-commercial, attribution).
# Safe to run again at any time - run it again on the day of the presentation.

set -euo pipefail

cd "$(dirname "$0")/.."

MVTEC_DIR="${MVTEC_DIR:-/workspaces/mvtec_anomaly_detection}"
API_IMAGE="anomaly-api:dev"
CATEGORIES=(bottle wood pill)

declare -A MVTEC_URLS=(
    [bottle]="https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz"
    [wood]="https://www.mydrive.ch/shares/150465/d5b4115b720cdb54d217e75636e6e374/download/420938383-1629960649/wood.tar.xz"
    [pill]="https://www.mydrive.ch/shares/150460/d4f1c04da67034ccc8d8b5fa3e73f244/download/420938129-1629960351/pill.tar.xz"
)

log() { printf '\n== %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Python helpers run in the API image: it has psycopg2 and reaches the
# published ports of the stack through the host network.
run_in_api_image() {
    docker run --rm --network host \
        --user "$(id -u):$(id -g)" \
        -v "$PWD":/work -w /work \
        "$@"
}

# ----------------------------------------------------------
log "1/6 MVTec AD data in $MVTEC_DIR"
# ----------------------------------------------------------

if ! command -v xz > /dev/null; then
    echo "xz is missing: installing xz-utils"
    sudo apt-get update -qq
    sudo apt-get install -y -qq xz-utils
fi

mkdir -p "$MVTEC_DIR"

for category in "${CATEGORIES[@]}"; do
    if [ -d "$MVTEC_DIR/$category/train/good" ]; then
        echo "$category: present"
        continue
    fi

    # Extract into a temporary folder first: an interrupted download never
    # leaves a half-filled category that a rerun would take as complete.
    partial="$MVTEC_DIR/.partial-$category"
    if [ -e "$partial" ]; then
        chmod -R u+w "$partial"
        rm -rf "$partial"
    fi
    mkdir -p "$partial"

    echo "$category: downloading"
    curl -fsSL "${MVTEC_URLS[$category]}" | tar -xJ -C "$partial"

    # The archives contain read-only folders; a folder can only be moved
    # (or later removed) when it is writable for its owner.
    chmod -R u+w "$partial"

    mv "$partial/$category" "$MVTEC_DIR/$category"
    rmdir "$partial"
done

# ----------------------------------------------------------
log "2/6 Docker Compose stack"
# ----------------------------------------------------------

# data/ is gitignored and bind-mounted into the API. Created here by the
# caller; otherwise Docker would create it as root.
mkdir -p data

docker compose up -d --build

# ----------------------------------------------------------
log "3/6 PostgreSQL metadata"
# ----------------------------------------------------------

postgres_ready=false
for _ in $(seq 1 60); do
    if docker compose exec -T postgres pg_isready -U anomaly_user -d anomaly_db > /dev/null 2>&1; then
        postgres_ready=true
        break
    fi
    sleep 2
done
[ "$postgres_ready" = true ] || fail "PostgreSQL is not ready after 2 minutes: docker compose logs postgres"

image_rows=$(
    docker compose exec -T postgres \
        psql -U anomaly_user -d anomaly_db -tAc "SELECT COUNT(*) FROM images" 2> /dev/null \
        || echo 0
)

if [ "${image_rows:-0}" -gt 0 ]; then
    echo "images table already has $image_rows rows: kept as it is (no reset)"
else
    run_in_api_image \
        -v "$MVTEC_DIR":/mvtec:ro \
        -e MVTEC_SOURCE_DIR=/mvtec \
        -e POSTGRES_HOST=127.0.0.1 \
        "$API_IMAGE" python database/load_data.py

    # load_data.py deletes and recreates data/. A running API container still
    # sees the deleted folder through its bind mount until it is recreated.
    docker compose up -d --force-recreate --no-deps api
fi

# ----------------------------------------------------------
log "4/6 Health checks"
# ----------------------------------------------------------

if ! docker compose exec -T streamlit \
    python -c "import socket; socket.create_connection(('postgres', 5432), 3)" > /dev/null 2>&1; then
    fail "containers cannot reach each other (Docker network). Check .devcontainer/devcontainer.json is in use."
fi
echo "containers reach each other: ok"

# The API exits when PostgreSQL was not reachable at its startup: start it again.
docker compose up -d

api_ready=false
for _ in $(seq 1 60); do
    if curl -fsS http://localhost:8000/health/db > /dev/null 2>&1; then
        api_ready=true
        break
    fi
    sleep 3
done
[ "$api_ready" = true ] || fail "API is not healthy after 3 minutes: docker compose logs api"
echo "API /health/db: ok"

# ----------------------------------------------------------
log "5/6 Champion models"
# ----------------------------------------------------------

run_in_api_image "$API_IMAGE" python scripts/demo_train.py

# ----------------------------------------------------------
log "6/6 Readiness check"
# ----------------------------------------------------------

run_in_api_image -e POSTGRES_HOST=127.0.0.1 "$API_IMAGE" python scripts/demo_check.py

# Both DAGs run every hour once unpaused (catchup is off, one run at a time).
# With all batches released the training DAG skips training; the drift DAG
# stores a drift report per category for Grafana.
for dag in anomaly_detection_incremental_training anomaly_detection_drift_monitoring; do
    docker compose exec -T airflow airflow dags unpause "$dag" > /dev/null
done
echo "Airflow DAGs: unpaused"

# ----------------------------------------------------------
log "Ready"
# ----------------------------------------------------------

# Airflow standalone creates the admin password at its first start.
airflow_password=$(
    docker compose exec -T airflow cat /opt/airflow/standalone_admin_password.txt 2> /dev/null \
        || echo "(not created yet - run this script again in a minute)"
)
printf 'Airflow login: admin / %s\n\n' "$airflow_password"

if [ -n "${CODESPACE_NAME:-}" ]; then
    base="https://${CODESPACE_NAME}"
    domain="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
    printf 'Streamlit   %s-8501.%s\n' "$base" "$domain"
    printf 'FastAPI     %s-8000.%s/docs\n' "$base" "$domain"
    printf 'MLflow      %s-5001.%s\n' "$base" "$domain"
    printf 'Airflow     %s-8081.%s\n' "$base" "$domain"
    printf 'Grafana     %s-3000.%s\n' "$base" "$domain"
    printf 'Prometheus  %s-9090.%s\n' "$base" "$domain"
else
    echo "Streamlit http://localhost:8501 - FastAPI http://localhost:8000/docs"
fi
