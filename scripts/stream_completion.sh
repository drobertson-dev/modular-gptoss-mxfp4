#!/usr/bin/env bash
# Stream and accumulate completions from the local MAX server.

set -euo pipefail

HOST="${HOST:-http://localhost:8000}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
PROMPT="${PROMPT:-Hello from mxfp4 this is a test (ping)}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

tmp_err="$(mktemp)"
trap 'rm -f "${tmp_err}"' EXIT

# Run the streaming pipeline; tolerate curl's EPIPE (23) if the reader finishes first.
set +o pipefail
curl -sN "${HOST}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "${PROMPT}"}],
  "max_tokens": ${MAX_TOKENS},
  "stream": true
}
EOF
  )" \
  2>"${tmp_err}" \
| sed -u 's/^data: //' \
| "${PYTHON_BIN}" -u <<'PY'
import json
import sys

buf = ""
for line in sys.stdin:
    line = line.strip()
    if not line or line == "[DONE]":
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    for choice in obj.get("choices", []):
        delta = choice.get("text", "")
        if delta:
            buf += delta
    sys.stdout.write("\r" + buf)
    sys.stdout.flush()
print()
PY
pipeline_status=("${PIPESTATUS[@]}")
set -o pipefail

curl_status="${pipeline_status[0]}"
sed_status="${pipeline_status[1]}"
py_status="${pipeline_status[2]}"

# Ignore curl's broken pipe (23) when the downstream consumer exits after [DONE].
if [[ "${curl_status}" != "0" && "${curl_status}" != "23" ]]; then
  echo "curl failed with status ${curl_status}" >&2
  cat "${tmp_err}" >&2 || true
  exit "${curl_status}"
fi

if [[ "${sed_status}" != "0" && "${sed_status}" != "141" ]]; then
  echo "sed failed with status ${sed_status}" >&2
  exit "${sed_status}"
fi

if [[ "${py_status}" != "0" ]]; then
  echo "python failed with status ${py_status}" >&2
  exit "${py_status}"
fi

# Show curl warnings only when not the harmless EPIPE.
if [[ -s "${tmp_err}" && "${curl_status}" == "23" ]]; then
  : # ignore
elif [[ -s "${tmp_err}" ]]; then
  cat "${tmp_err}" >&2 || true
fi
