#!/bin/sh
# If the mounted model volume has no artifacts yet (fresh clean-machine
# start), bootstrap them once so the API has something to load. On a real
# deployment this volume is populated by the artifact store / DVC pull
# instead, and this step is a no-op.
set -e

if [ ! -f "/srv/data/models/model.joblib" ]; then
  echo "No model artifact found at data/models/model.joblib — bootstrapping..."
  python scripts/bootstrap_train.py
fi

exec "$@"
