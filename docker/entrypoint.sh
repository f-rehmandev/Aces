#!/usr/bin/env bash
# ============================================================================
# ACES container entrypoint
#
# Modes (first argument):
#   api     — FastAPI server with in-process worker pool + scheduler
#   worker  — worker-only (no HTTP), consumes the durable queue
#   shell   — debug shell (drops into bash)
#
# Any remaining arguments after the mode are passed through.
# ============================================================================
set -euo pipefail

MODE="${1:-api}"
shift || true

case "$MODE" in
  api)
    echo "[entrypoint] starting ACES API"
    echo "[entrypoint] python: $(python --version 2>&1)"
    # uvicorn only accepts lowercase log levels; .env may carry "INFO"
    LOG_LEVEL="$(echo "${ACES_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
    exec uvicorn src.api.production:create_app \
        --factory \
        --host 0.0.0.0 \
        --port "${ACES_PORT:-8000}" \
        --log-level "$LOG_LEVEL" \
        "$@"
    ;;

  worker)
    echo "[entrypoint] starting ACES worker-only process"
    exec python -m src.jobs.worker_only "$@"
    ;;

  shell|bash)
    echo "[entrypoint] dropping into interactive shell"
    exec /bin/bash "$@"
    ;;

  *)
    echo "[entrypoint] unknown mode: $MODE" >&2
    echo "usage: entrypoint.sh [api|worker|shell] [args...]" >&2
    exit 2
    ;;
esac