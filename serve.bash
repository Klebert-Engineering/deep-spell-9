#!/usr/bin/env bash

set -e

export MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

used_server_port="${1:-8091}"

# Determine python virtual environment
VIRTUAL_ENV_ARG=()
if [[ -d "$VIRTUAL_ENV" ]]; then
  VIRTUAL_ENV_ARG=(--virtualenv "$VIRTUAL_ENV")
fi

# Hand over to UWSGI HTTP server
cd "$MY_DIR"
exec uwsgi \
    "${VIRTUAL_ENV_ARG[@]}" \
    --processes 4 \
    --enable-threads \
    --http ":$used_server_port" \
    --wsgi-file "service.py" \
    --callable app
