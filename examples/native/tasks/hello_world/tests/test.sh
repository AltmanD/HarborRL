#!/bin/sh
set -eu

reward=0
if [ "$(cat /hello 2>/dev/null || true)" = "ok" ]; then
    reward=1
fi
mkdir -p /logs/verifier
printf '{"reward":%s}\n' "$reward" >/logs/verifier/reward.json
