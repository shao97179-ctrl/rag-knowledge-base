from rag_learning.logging_config import configure_logging

# 测试期间只保留 WARNING，避免 pymilvus / transformers 的日志淹没断言输出
configure_logging("WARNING")
