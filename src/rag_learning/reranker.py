"""BGE reranker（cross-encoder）二阶段精排：对 RRF 融合后的候选重排序。

双塔召回分不开「镜像文档」（工资账户转储蓄 vs 储蓄账户转工资）这类
表面相似、语义相反的问题，cross-encoder 把 query 和文档拼在一起打分，
是补这个短板的标准做法。模型目录缺失时自动跳过，检索退回纯 RRF 排序。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .embedder import _DTYPE_MAP

logger = logging.getLogger(__name__)


class BgeReranker:
    """bge-reranker-v2-m3 封装：懒加载、缺模型跳过、OOM 降 CPU。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        dtype: str = "float16",
        batch_size: int = 8,
        max_length: int = 512,
        enabled: bool = True,
    ) -> None:
        """只保存参数；enabled=False 或模型目录不存在时 loaded 永远为 False。"""
        self.model_path = Path(model_path)
        self.requested_device = device
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.enabled = enabled and self.model_path.exists()
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.dtype = torch.float32
        self._loaded = False
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        """同一 key 的 WARNING 只输出一次。"""
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message)

    def _ensure_loaded(self) -> bool:
        """懒加载模型；不可用（未启用/缺目录）返回 False，调用方跳过精排。"""
        if self._loaded:
            return True
        if not self.enabled:
            self._warn_once(
                "disabled",
                f"reranker 未启用（enabled=False 或目录不存在：{self.model_path}），"
                "检索使用纯 RRF 排序。下载模型后放到该目录即可自动启用",
            )
            return False
        with self._lock:
            if self._loaded:
                return True
            if self.requested_device == "cpu":
                self.device = "cpu"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.dtype = _DTYPE_MAP[self.dtype_name]
            if self.device == "cpu" and self.dtype == torch.float16:
                self._warn_once("cpu_fp16", "reranker 在 CPU 上运行，自动改用 float32")
                self.dtype = torch.float32

            logger.info("加载 reranker：%s（device=%s, dtype=%s）", self.model_path, self.device, self.dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            # safetensors / bin 都接受，按目录里实际存在的权重加载
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                dtype=self.dtype,
            )
            self.model.eval().to(self.device)
            self._loaded = True
            logger.info("reranker 加载完成")
        return True

    @torch.inference_mode()
    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """一批 (query, document) 对的相关性分数（sigmoid 到 0~1）。"""
        inputs = self.tokenizer(
            [query for query, _ in pairs],
            [document for _, document in pairs],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits.view(-1).float()
        return torch.sigmoid(logits).tolist()

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回每篇文档与 query 的相关性分数，顺序与输入一致。"""
        if not documents:
            return []
        if not self._ensure_loaded():
            return []
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = [
                (query, document[:2000] if document else " ")
                for document in documents[start:start + self.batch_size]
            ]
            try:
                with self._infer_lock:
                    scores.extend(self._score_batch(batch))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._warn_once("oom_cpu", "reranker 显存不足，切换到 CPU 推理")
                self.device = "cpu"
                if self.model is not None:
                    self.model.to("cpu")
                with self._infer_lock:
                    scores.extend(self._score_batch(batch))
        return scores

    @property
    def loaded(self) -> bool:
        """模型是否已加载；未启用的 reranker 恒为 False。"""
        return self._loaded

    @property
    def available(self) -> bool:
        """是否配置了可用模型（不触发加载）。"""
        return self.enabled

    @property
    def memory_mb(self) -> float | None:
        """当前 CUDA 显存占用（MiB）；CPU 模式返回 None。"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return None
        return round(torch.cuda.memory_allocated() / 1024 ** 2, 1)

    def close(self) -> None:
        """释放模型引用并清空显存缓存。"""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_RERANKER: BgeReranker | None = None


def get_reranker(settings) -> BgeReranker:
    """进程内单例；配置关掉时返回一个永远不加载的实例。"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = BgeReranker(
            model_path=settings.reranker_path,
            device=settings.embed_device,
            dtype=settings.embed_dtype,
            batch_size=settings.embed_batch_size,
            enabled=settings.rerank_enabled,
        )
    return _RERANKER
