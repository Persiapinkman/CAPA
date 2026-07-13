import requests
import logging
from typing import Dict, Any
from src.core.config import get_settings
import time

settings = get_settings()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGApiTester:
    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = f"http://localhost:{settings.PORT}{settings.API_V1_STR}"
        self.base_url = base_url
        self.chunking_url = f"{base_url}/rag/doc_engine/chunking_embedding"
        self.query_url = f"{base_url}/rag/chat_engine/query"

    def _make_request(self, url: str, data: Dict[str, Any], operation: str) -> Dict:
        """发送请求并处理响应"""
        try:
            response = requests.post(url, json=data)
            logger.info(f"{operation} - 状态码: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"{operation} 失败: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"{operation} 请求异常: {str(e)}")
            return {}

    def get_test_chunking_data(self) -> Dict[str, Any]:
        """准备文档处理测试数据"""
        return {
            "text": "这是一个用于验证 chunking_embedding 接口可用性的测试文档。\n\n文本包含多段内容，用于触发分块和向量化流程。",
            "input_type": "raw",
            "doc_id": "_test_doc_raw_001",
            "doc_name": "data_source_smoke_test.txt",
            "embedding_models": settings.DEFAULT_EMBEDDING_MODELS
        }

    def get_test_retrieving_data(self) -> Dict[str, Any]:
        """准备检索测试数据"""
        return {
            "query": "如何在软件中切换到cobas link界面？",
            "top_k": 5,
            "uri": settings.TEST_VECTOR_DB_URI,
            "collection_name": settings.TEST_COLLECTION_NAME,
            "embedding_models": settings.DEFAULT_EMBEDDING_MODELS
        }

    def test_chunking(self) -> bool:
        """测试文档处理接口"""
        logger.info("Starting document processing API test...")
        data = self.get_test_chunking_data()
        start_time = time.time()
        result = self._make_request(self.chunking_url, data, "Document Processing")

        if result:
            logger.info(f"Document processing successful! Returned node count: {len(result.get('index_nodes', []))}, time: {time.time() - start_time:.2f}s")
            return True
        return False

    def test_query(self) -> bool:
        """测试问答接口"""
        logger.info("Starting RAG chat API test...")
        data = self.get_test_retrieving_data()
        start_time = time.time()
        result = self._make_request(self.query_url, data, "RAG Chat")

        if result:
            references = result.get("reference", [])
            timings = result.get("timings", {})
            logger.info(
                f"RAG chat successful! Returned reference count: {len(references)}, time: {time.time() - start_time:.2f}s, timings={timings}"
            )
            if references:
                logger.info(f"First reference: {references[0]}")
            return True
        return False

    def run_all_tests(self) -> None:
        """运行所有测试"""
        test_results = {
            "Document Processing Test": self.test_chunking(),
            "RAG Chat Test": self.test_query(),
        }

        logger.info("\nTest Results Summary:")
        for test_name, result in test_results.items():
            logger.info(f"{test_name}: {'Passed' if result else 'Failed'}")

def main():
    # 从配置中获取基础URL
    base_url = f"http://localhost:{settings.PORT}{settings.API_V1_STR}"

    tester = RAGApiTester(base_url)
    tester.run_all_tests()

if __name__ == "__main__":
    main()
