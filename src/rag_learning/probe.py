"""probe 自检：模型、向量库、LLM 三项逐一验证。"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch


def run_probe(live: bool = False) -> int:
    """依次验证 BGE-M3、milvus-lite、LLM，成功返回 0，失败返回 1。

    不写死任何 data 目录：向量库用临时文件，跑完自动清理。
    """
    from .config import get_settings
    from .embedder import BgeM3Embedder
    from .llm import LLMClient
    from .vector_store import MilvusStore

    settings = get_settings()
    print("=" * 64)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    try:
        import pymilvus
        import transformers

        print(f"transformers={transformers.__version__} pymilvus={pymilvus.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"依赖检查失败：{exc}")
        return 1

    texts = [
        "数据库连接池耗尽时，先检查 max_connections 和慢查询。",
        "P0 故障需要在 5 分钟内响应并通知值班经理。",
        "员工的年假可以跨年顺延，但最多保留到次年三月。",
    ]
    # 直接用 BgeM3Embedder 而不是 get_embedder，避免污染进程内单例
    embedder = BgeM3Embedder(
        model_path=settings.bge_m3_path,
        device=settings.embed_device,
        dtype=settings.embed_dtype,
        batch_size=settings.embed_batch_size,
        max_length=settings.embed_max_length,
        query_max_length=settings.query_max_length,
    )
    try:
        started = time.perf_counter()
        dense, sparse = embedder.encode(texts)
        elapsed = time.perf_counter() - started
        # dense 已经 L2 归一化，范数应该接近 1，可以顺手验证实现正确性
        norms = [round(sum(value * value for value in vector) ** 0.5, 6) for vector in dense]
        print(f"\n[1/3] BGE-M3: dense={len(dense[0])} 维 范数={norms} 耗时={elapsed:.2f}s")
        for index, vector in enumerate(sparse[:2]):
            # 把稀疏向量的 token id 解码成文字，肉眼确认权重是否落在关键词上
            top = sorted(vector.items(), key=lambda item: item[1], reverse=True)[:8]
            words = [
                (embedder.tokenizer.decode([token_id]), round(weight, 3))
                for token_id, weight in top
            ]
            print(f"  稀疏向量[{index}] 长度={len(vector)} top={words}")
        if embedder.memory_mb is not None:
            print(f"  显存占用={embedder.memory_mb} MiB  device={embedder.device}")

        with tempfile.TemporaryDirectory(prefix="rag_probe_") as tmp:
            # 用临时库验证建表、写入、dense/sparse 检索全链路
            store = MilvusStore(
                uri=str(Path(tmp) / "probe.db"),
                collection="probe",
                dimension=embedder.DIMENSION,
            )
            try:
                store.ensure_collection()
                store.upsert(
                    [
                        {
                            "pk": f"probe-{index}",
                            "doc_id": f"probe-doc-{index}",
                            "chunk_index": index,
                            "parent_id": f"probe-parent-{index}",
                            "child_text": text,
                            "parent_text": text,
                            "heading": "probe",
                            "title": "probe",
                            "source_path": f"probe/{index}.txt",
                            "department": "general",
                            "sensitivity": "public",
                            "doc_format": "text",
                            "content_hash": "probe",
                            "dense": dense[index],
                            "sparse": sparse[index],
                        }
                        for index, text in enumerate(texts)
                    ]
                )
                dense_hits = store.search_dense(dense[0], 'sensitivity in ["public"]', 2)
                sparse_hits = store.search_sparse(sparse[0], 'sensitivity in ["public"]', 2)
                print(
                    f"\n[2/3] milvus-lite: rows={store.count()} "
                    f"dense_top={[hit['source_path'] for hit in dense_hits]} "
                    f"sparse_top={[hit['source_path'] for hit in sparse_hits]}"
                )
            finally:
                store.close()
    except Exception as exc:  # noqa: BLE001
        # 任何一步失败都直接打出来，方便定位是环境问题还是代码问题
        print(f"\nprobe 失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        embedder.close()

    llm = LLMClient(settings)
    print(
        f"\n[3/3] LLM: provider={settings.llm_provider} model={settings.llm_model} "
        f"available={llm.available}"
    )
    if live:
        # --live 会真实产生一次调用费用，默认不跑
        if not llm.available:
            print("  没有配置 API Key，跳过真实调用")
        else:
            answer = llm.chat("你是一个简洁的助手。", "用一句话说明什么是 RAG。")
            print(f"  LLM 回答：{answer or '调用失败'}")
    print("\nprobe 完成")
    return 0
