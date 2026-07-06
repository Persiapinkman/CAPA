#!/usr/bin/env bash
set -euo pipefail

# Call rag-retrieve-answer (HTTP RAG service). Prints the answer on stdout and
# writes the full JSON response under results/ by default.
#
# Usage:
#   bash examples/test_rag.sh [query] [out_json] [rag_url]
#
# Examples:
#   bash examples/test_rag.sh
#   bash examples/test_rag.sh "你的问题"
#   bash examples/test_rag.sh "你的问题" results/my_rag/resp.json
#   bash examples/test_rag.sh "你的问题" results/my_rag/resp.json "http://host:6062/api/v1/playbook/query"
#
# Environment (optional, see skill SKILL.md):
#   RAG_TEST_QUERY     default question if [query] omitted
#   RAG_QUERY_URL      base URL if [rag_url] omitted
#   RAG_URI, RAG_COLLECTION_NAME, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD
#
# If you see "Connection refused" (Errno 111): nothing is listening on the default host:port.
# Start the RAG service or set RAG_QUERY_URL, e.g.:
#   export RAG_QUERY_URL='http://127.0.0.1:6062/api/v1/playbook/query'
#   bash examples/test_rag.sh

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage:"
  echo "  bash examples/test_rag.sh [query] [out_json] [rag_url]"
  echo ""
  echo "Defaults:"
  echo "  query:    RAG_TEST_QUERY env, or built-in sample question about safety_rope"
  echo "  out_json: results/rag_test/response.json"
  echo "  rag_url:  RAG_QUERY_URL env, or run_rag.py built-in default"
  echo ""
  echo "Other RAG_* env vars are passed through to run_rag.py (see skills/rag-retrieve-answer/SKILL.md)."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_QUERY='safety_rope v0.2.1 模型检测的是什么目标？'
if [ -n "${1:-}" ]; then
  QUERY="$1"
elif [ -n "${RAG_TEST_QUERY:-}" ]; then
  QUERY="${RAG_TEST_QUERY}"
else
  QUERY="${DEFAULT_QUERY}"
fi

OUT_JSON="${2:-results/rag_test/response.json}"
RAG_URL="${3:-${RAG_QUERY_URL:-}}"

cd "${ROOT}"

mkdir -p "$(dirname "${OUT_JSON}")"

cmd=(
  python3
  skills/rag-retrieve-answer/scripts/run_rag.py
  --query "${QUERY}"
  --out "${OUT_JSON}"
)
if [ -n "${RAG_URL}" ]; then
  cmd+=( --base-url "${RAG_URL}" )
fi

echo "Query:  ${QUERY}"
echo "Out:    ${OUT_JSON}"
echo "--------------------------------------------------"

"${cmd[@]}"

echo "--------------------------------------------------"
echo "Full JSON saved to: ${OUT_JSON}"
