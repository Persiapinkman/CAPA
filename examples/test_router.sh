#!/usr/bin/env bash
set -euo pipefail

# Test demo/router.py via examples/test_router.py
#
# Usage:
#   bash examples/test_router.sh [case]
#
# Cases:
#   rag        (default) 走知识库问答（不传图）
#   flux       基于图片生成图片，默认 3 张
#   detect     开集目标检测
#   pipeline   完整目标检测评测流水线
#   rag        走知识库问答（不传图）

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash examples/test_router.sh [case]"
  echo "Cases: flux | detect | pipeline | rag"
  exit 0
fi

CASE="${1:-rag}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

IMAGE="examples/images/fisherman.jpg"
TEXT=""

case "${CASE}" in
  flux)
    TEXT="请基于这张图生成3张新图，保持监控视角但场景细节明显变化"
    ;;
  detect)
    TEXT="请检测这张图中的钓鱼人物并框出来"
    ;;
  pipeline)
    TEXT="请做完整评测：生成扩展图、检测并比较模型准确率"
    ;;
  rag)
    TEXT="safety_rope v0.2.1 模型检测的是什么目标？"
    IMAGE=""
    ;;
  *)
    echo "Unknown case: ${CASE}"
    echo "Use one of: flux | detect | pipeline | rag"
    exit 2
    ;;
esac

CMD=(python3 examples/test_router.py --text "${TEXT}")
if [ -n "${IMAGE}" ]; then
  CMD+=(--image "${IMAGE}")
fi

echo "Case:  ${CASE}"
echo "Text:  ${TEXT}"
if [ -n "${IMAGE}" ]; then
  echo "Image: ${IMAGE}"
else
  echo "Image: (none)"
fi
echo "----------------------------------------"
"${CMD[@]}"

