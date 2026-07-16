#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Open SSH forwards for the production GBrain and ACE RAG services.

Environment overrides:
  CAPA_RAG_SSH_TARGET   SSH target (default: linshihao@10.111.32.254)
  CAPA_SOCKS_PROXY      local SOCKS address (default: 127.0.0.1:8888)
  CAPA_GBRAIN_PORT      local GBrain port (default: 6061)
  CAPA_ACE_PORT         local ACE port (default: 6062)

The SSH password is requested interactively and is never stored by this script.
EOF
  exit 0
fi

for command in ssh nc; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Required command is missing: %s\n' "$command" >&2
    exit 1
  fi
done

target="${CAPA_RAG_SSH_TARGET:-linshihao@10.111.32.254}"
socks_proxy="${CAPA_SOCKS_PROXY:-127.0.0.1:8888}"
gbrain_port="${CAPA_GBRAIN_PORT:-6061}"
ace_port="${CAPA_ACE_PORT:-6062}"

printf 'Opening RAG forwards through %s; keep this terminal running.\n' "$target"
exec ssh -N \
  -o "ProxyCommand=nc -x ${socks_proxy} -X 5 %h %p" \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "${gbrain_port}:127.0.0.1:6061" \
  -L "${ace_port}:127.0.0.1:6062" \
  "$target"
