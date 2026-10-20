"""Milvus 存储层：确定性主键、dense/sparse 双字段、ACL filter、尾巴删除。"""

from __future__ import annotations

import logging
from pathlib import Path

from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)

# 检索时需要的标量字段；向量本身不返回，节省传输和内存
OUTPUT_FIELDS = [
    "doc_id",
    "chunk_index",
    "parent_id",
    "child_text",
    "parent_text",
    "heading",
    "title",
    "source_path",
    "department",
    "sensitivity",
    "doc_format",
]


def _quote(value: str) -> str:
    """转义表达式里的字符串常量，避免引号注入破坏 filter。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class MilvusStore:
    """Milvus 封装：schema、写入、删除、双路检索都收敛在这个类里。"""

    def __init__(self, uri: str, collection: str, dimension: int = 1024) -> None:
        """只记录连接参数，客户端延迟到第一次使用时创建。"""
        self.uri = uri
        self.collection = collection
        self.dimension = dimension
        self._client: MilvusClient | None = None

    # ---------- 连接与 schema ----------

    @property
    def client(self) -> MilvusClient:
        """懒创建 Milvus 客户端；本地 uri 会顺便建好父目录。"""
        if self._client is None:
            if not self.uri.startswith(("http://", "https://")):
                Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
            logger.info("连接 Milvus：%s", self.uri)
            self._client = MilvusClient(uri=self.uri)
        return self._client

    def ensure_collection(self) -> None:
        """集合不存在时创建 schema 和索引，并加载到内存；已存在则直接返回。"""
        if self.client.has_collection(self.collection):
            return
        logger.info("创建集合：%s（dim=%s）", self.collection, self.dimension)
        # auto_id=False：主键由 chunker 的 sha1 决定，保证 upsert 幂等
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        # 注意：VARCHAR 的 max_length 是字节数，中文一个字占 3 字节
        schema.add_field("child_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("heading", DataType.VARCHAR, max_length=2048)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("source_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("department", DataType.VARCHAR, max_length=32)
        schema.add_field("sensitivity", DataType.VARCHAR, max_length=16)
        schema.add_field("doc_format", DataType.VARCHAR, max_length=16)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        # 数据量小，dense 用 FLAT 精确检索；两者都用内积（向量已 L2 归一化）
        index_params.add_index(field_name="dense", index_type="FLAT", metric_type="IP")
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name=self.collection)

    def recreate(self) -> None:
        """--force 重建：drop 后验证是否残留，必要时物理删库文件。"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(collection_name=self.collection)
        self.ensure_collection()
        if self.count() == 0:
            return
        # milvus-lite 的 drop 偶发不彻底，只能关连接后删 db 文件
        logger.warning("milvus-lite drop 后仍有残留数据，物理删除数据库文件")
        self.close()
        Path(self.uri).unlink(missing_ok=True)
        Path(f"{self.uri}.lock").unlink(missing_ok=True)
        self.ensure_collection()

    def close(self) -> None:
        """关闭客户端连接；下一次访问 self.client 会重新建立。"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def load(self) -> None:
        """显式加载集合，API 启动时可以提前预热。"""
        self.ensure_collection()
        self.client.load_collection(collection_name=self.collection)

    def _ensure_loaded(self) -> None:
        """跨进程使用时集合可能处于 released，查询/删除前必须重新 load。"""
        self.ensure_collection()
        self.client.load_collection(collection_name=self.collection)

    # ---------- 写入 ----------

    def upsert(self, rows: list[dict]) -> None:
        """按主键 upsert，同一 pk 重复写入不会产生重复行。"""
        if not rows:
            return
        self._ensure_loaded()
        self.client.upsert(collection_name=self.collection, data=rows)

    def delete_doc(self, doc_id: str) -> None:
        """删除某篇文档的所有 chunk（文档被删除或内容变空时使用）。"""
        if not self.client.has_collection(self.collection):
            return
        self._ensure_loaded()
        self.client.delete(collection_name=self.collection, filter=f'doc_id == "{_quote(doc_id)}"')

    def delete_tail(self, doc_id: str, keep_count: int) -> None:
        """文档变短时删除 chunk_index >= keep_count 的残留块。

        必须在 upsert 之前调用，否则新数据会被误删。
        """
        if not self.client.has_collection(self.collection):
            return
        self._ensure_loaded()
        expression = f'doc_id == "{_quote(doc_id)}" and chunk_index >= {int(keep_count)}'
        self.client.delete(collection_name=self.collection, filter=expression)

    # ---------- 查询 ----------

    def _search(self, data, anns_field: str, expression: str, limit: int) -> list[dict]:
        """dense / sparse 共用的检索实现，把 hit 整理成普通 dict。"""
        self._ensure_loaded()
        results = self.client.search(
            collection_name=self.collection,
            data=[data],
            anns_field=anns_field,
            limit=limit,
            # ACL 条件在检索层执行，无权内容不会进入候选集
            filter=expression,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "IP", "params": {}},
        )
        hits: list[dict] = []
        for hit in results[0]:
            entity = dict(hit.get("entity") or {})
            # 自定义主键时 hit 的字段名是 pk（不是 id），拿不到再退回属性访问
            try:
                pk_value = hit["pk"]
            except Exception:  # noqa: BLE001
                pk_value = getattr(hit, "id", "")
            entity["pk"] = str(pk_value)
            entity["distance"] = float(hit["distance"])
            hits.append(entity)
        return hits

    def search_dense(self, vector: list[float], expression: str, limit: int) -> list[dict]:
        """稠密向量检索：语义相似度，按内积从大到小。"""
        return self._search(vector, "dense", expression, limit)

    def search_sparse(self, vector: dict[int, float], expression: str, limit: int) -> list[dict]:
        """稀疏向量检索：关键词权重匹配，适合专有名词和编号。"""
        return self._search(vector, "sparse", expression, limit)

    def count(self) -> int:
        """集合总行数；集合不存在时返回 0。"""
        if not self.client.has_collection(self.collection):
            return 0
        self._ensure_loaded()
        result = self.client.query(
            collection_name=self.collection,
            filter="pk != ''",
            output_fields=["count(*)"],
        )
        if not result:
            return 0
        return int(result[0].get("count(*)", 0))

    def stats(self) -> dict:
        """健康检查 / 状态展示用的集合摘要。"""
        return {
            "uri": self.uri,
            "collection": self.collection,
            "exists": self.client.has_collection(self.collection),
            "rows": self.count(),
            "dimension": self.dimension,
        }
