"""BGE-M3 编码器：裸 transformers 实现 dense + sparse 两种表示。

dense：最后一层 [CLS] 隐状态 -> L2 归一化，1024 维。
sparse：隐状态 @ sparse_linear 权重 -> ReLU -> 同一 token 取最大权重。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# 配置里的字符串精度名到 torch 类型的映射
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class BgeM3Embedder:
    """BGE-M3 编码器，一次前向同时产出 dense 和 sparse 两种向量。"""

    # dense 向量维度固定 1024；Milvus schema 和测试都依赖这个常量
    DIMENSION = 1024

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        dtype: str = "float16",
        batch_size: int = 8,
        max_length: int = 1024,
        query_max_length: int = 256,
    ) -> None:
        """只保存参数，真正加载模型延迟到第一次 encode（懒加载）。"""
        self.model_path = Path(model_path)
        self.requested_device = device
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_max_length = query_max_length

        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.dtype = torch.float32
        self._sparse_weight: torch.Tensor | None = None
        self._sparse_bias: torch.Tensor | None = None
        self._special_ids: set[int] = set()
        self._loaded = False
        # 双重检查锁：多线程首次 encode 时只加载一次模型
        self._lock = threading.Lock()
        # 已经打过的告警 key，避免显存不足时刷屏
        self._warned: set[str] = set()

    # ---------- 加载 ----------

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        """同一 key 的 WARNING 只输出一次。"""
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    def _resolve_device(self) -> str:
        """按配置和实际硬件决定设备：auto 优先 CUDA，显式 cuda 不可用则直接报错。"""
        if self.requested_device == "cpu":
            return "cpu"
        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("配置了 RAG_EMBED_DEVICE=cuda，但当前环境没有可用 CUDA")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        """懒加载 tokenizer、模型权重和 sparse_linear；重复调用是空操作。"""
        if self._loaded:
            return
        with self._lock:
            # 拿到锁后二次检查：可能已被另一个线程加载完成
            if self._loaded:
                return
            if not self.model_path.exists():
                raise FileNotFoundError(f"BGE-M3 模型目录不存在：{self.model_path}")

            self.device = self._resolve_device()
            self.dtype = _DTYPE_MAP[self.dtype_name]
            # CPU 上 fp16 推理极慢甚至不支持，自动降为 fp32
            if self.device == "cpu" and self.dtype == torch.float16:
                self._warn_once("cpu_fp16", "CPU 不支持 fp16 推理，自动改用 float32")
                self.dtype = torch.float32

            logger.info("加载 BGE-M3：%s（device=%s, dtype=%s）", self.model_path, self.device, self.dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            # 模型目录只有 pytorch_model.bin，必须显式关闭 safetensors
            self.model = AutoModel.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                use_safetensors=False,
            )
            self.model.eval().to(self.device)
            # 特殊 token（CLS/SEP/PAD 等）不参与稀疏权重统计
            self._special_ids = set(self.tokenizer.all_special_ids)

            # sparse_linear.pt 是 OrderedDict（fp16 weight + bias），不是裸 tensor；
            # 在 fp32 下计算，避免 fp16 下溢导致权重全 0
            state = torch.load(
                self.model_path / "sparse_linear.pt",
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict):
                weight = state.get("weight", state.get("sparse_linear.weight"))
                bias = state.get("bias", state.get("sparse_linear.bias"))
            else:
                weight, bias = state, None
            if weight is None:
                raise RuntimeError("sparse_linear.pt 中找不到 weight")
            # 统一 reshape 成 (1, 1024)，方便后面用矩阵乘算 token 权重
            weight = weight.detach().float().reshape(1, self.DIMENSION)
            self._sparse_weight = weight.to(self.device)
            self._sparse_bias = None if bias is None else bias.detach().float().reshape(-1)[:1].to(self.device)
            self._loaded = True
            logger.info("BGE-M3 加载完成")

    # ---------- 编码 ----------

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str], is_query: bool) -> tuple[list[list[float]], list[dict[int, float]]]:
        """对一批文本做一次前向，返回 dense 列表和 sparse 字典列表。

        is_query 决定截断长度：查询更短（默认 256），文档更长（默认 1024）。
        """
        max_length = self.query_max_length if is_query else self.max_length
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        assert self.model is not None
        hidden = self.model(**inputs).last_hidden_state

        # dense：取第 0 个 token（[CLS]）的隐状态并做 L2 归一化；
        # 转 fp32 再归一化，避免 fp16 精度问题
        dense = F.normalize(hidden[:, 0].float(), p=2, dim=-1).cpu().tolist()

        # sparse：每个 token 的隐状态过一个 1024->1 的线性层，ReLU 得到重要性
        assert self._sparse_weight is not None
        weights = hidden.float() @ self._sparse_weight.T
        if self._sparse_bias is not None:
            weights = weights + self._sparse_bias
        weights = torch.relu(weights).squeeze(-1)

        mask = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"]
        sparse_vectors: list[dict[int, float]] = []
        for row in range(len(texts)):
            # 只统计真实 token，跳过 padding
            row_ids = input_ids[row][mask[row]].tolist()
            row_weights = weights[row][mask[row]].tolist()
            aggregated: dict[int, float] = {}
            for token_id, token_weight in zip(row_ids, row_weights):
                if token_id in self._special_ids or token_weight <= 1e-6:
                    continue
                # 同一个 token 多次出现取最大权重，与 BGE-M3 官方实现一致
                if token_weight > aggregated.get(token_id, 0.0):
                    aggregated[token_id] = float(token_weight)
            # Milvus 不接受空稀疏向量，用极小权重占位
            if not aggregated:
                aggregated = {0: 1e-7}
            sparse_vectors.append(aggregated)
        return dense, sparse_vectors

    def _encode_with_fallback(
        self, texts: list[str], is_query: bool
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """显存不足时的三级降级：拆 batch -> 转 CPU -> 原样抛出。"""
        try:
            return self._encode_batch(texts, is_query)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if self.device == "cuda" and len(texts) > 1:
                # 二分拆 batch 递归重试，并下调后续 batch_size
                middle = len(texts) // 2
                self.batch_size = max(1, min(self.batch_size, len(texts) // 2))
                self._warn_once("oom", "显存不足，自动拆小 batch（当前 batch=%s）", len(texts))
                dense_a, sparse_a = self._encode_with_fallback(texts[:middle], is_query)
                dense_b, sparse_b = self._encode_with_fallback(texts[middle:], is_query)
                return dense_a + dense_b, sparse_a + sparse_b
            if self.device == "cuda":
                # 单条也放不下：整个模型搬到 CPU，后续查询都在 CPU 上跑
                self._warn_once("cpu_fallback", "显存不足且无法再拆 batch，切换到 CPU 推理")
                self.model.to("cpu")  # type: ignore[union-attr]
                if self._sparse_weight is not None:
                    self._sparse_weight = self._sparse_weight.cpu()
                if self._sparse_bias is not None:
                    self._sparse_bias = self._sparse_bias.cpu()
                self.device = "cpu"
                torch.cuda.empty_cache()
                return self._encode_batch(texts, is_query)
            raise

    def encode(
        self, texts: list[str], is_query: bool = False
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """返回 (dense_vectors, sparse_vectors)。空文本用空格占位。"""
        if not texts:
            return [], []
        self._ensure_loaded()
        # 空字符串会让 tokenizer 产出全 padding，用空格占位更稳定
        normalized = [text if text and text.strip() else " " for text in texts]
        dense_vectors: list[list[float]] = []
        sparse_vectors: list[dict[int, float]] = []
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start:start + self.batch_size]
            dense, sparse = self._encode_with_fallback(batch, is_query)
            dense_vectors.extend(dense)
            sparse_vectors.extend(sparse)
        return dense_vectors, sparse_vectors

    def encode_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        """单条查询编码，使用更短的 max_length 降低延迟。"""
        dense, sparse = self.encode([query], is_query=True)
        return dense[0], sparse[0]

    # ---------- 观测 ----------

    @property
    def loaded(self) -> bool:
        """模型是否已经加载，供 health / probe 展示。"""
        return self._loaded

    @property
    def memory_mb(self) -> float | None:
        """当前 CUDA 显存占用（MiB）；CPU 模式返回 None。"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return None
        return round(torch.cuda.memory_allocated() / 1024 ** 2, 1)

    def close(self) -> None:
        """释放模型引用并清空显存缓存，长驻进程退出前调用。"""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_EMBEDDER: BgeM3Embedder | None = None


def get_embedder(settings) -> BgeM3Embedder:
    """进程内单例，避免重复加载 2.2GB 模型。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = BgeM3Embedder(
            model_path=settings.bge_m3_path,
            device=settings.embed_device,
            dtype=settings.embed_dtype,
            batch_size=settings.embed_batch_size,
            max_length=settings.embed_max_length,
            query_max_length=settings.query_max_length,
        )
    return _EMBEDDER
