export http_proxy=socks5h://127.0.0.1:8888
export https_proxy=socks5h://127.0.0.1:8888
export all_proxy=socks5h://127.0.0.1:8888
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export ALL_PROXY="$all_proxy"
export no_proxy=127.0.0.1,localhost
export NO_PROXY="$no_proxy"

# Company model services reached through the local SOCKS proxy.
export DEMO_LLM_API_BASE=http://10.111.32.253:8000/v1
export DEMO_REX_BASE_URL=http://10.111.32.253:8000/v1
export DEMO_QWEN_DETECTION_URL=http://10.111.32.254:9012/v1

# RAG is exposed locally by pipelines/demo/open_rag_tunnel.sh.
export DEMO_RAG_API_MODE=unified
export GBRAIN_RAG_BASE_URL=http://127.0.0.1:6061/api/v1/rag
export RAG_BASE_URL=http://127.0.0.1:6062/api/v1/playbook/query
