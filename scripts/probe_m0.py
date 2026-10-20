"""M0 技术验证脚本：BGE-M3 加载 + dense/sparse 编码 + milvus-lite 稀疏检索。

这个脚本不依赖项目包，先把三个高风险点跑通：
1. transformers 5.3 能否加载本地 pytorch_model.bin；
2. sparse_linear.pt 的真实结构，以及稀疏权重能否算出来；
3. milvus-lite 能否建稀疏索引并做 dense/sparse 双路检索。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

# tokenizer 的多线程会在某些环境里刷 warning，脚本里直接关掉
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = Path(os.environ.get("RAG_BGE_M3_PATH", "BGE-M3"))
TEXTS = [
    "数据库连接池耗尽时，先检查 max_connections 和慢查询。",
    "P0 故障需要在 5 分钟内响应并通知值班经理。",
    "员工的年假可以跨年顺延，但最多保留到次年三月。",
]


def inspect_sparse_linear() -> torch.Tensor:
    """检查 sparse_linear.pt 的真实结构，返回 fp32 的 (1, 1024) 权重和 bias。

    预期是 OrderedDict（fp16 weight + bias），但先打印结构再断言，
    这样结构变化时能直接从输出看出原因。
    """
    print("\n=== 1. sparse_linear.pt 结构 ===")
    obj = torch.load(MODEL_PATH / "sparse_linear.pt", map_location="cpu", weights_only=True)
    print("type:", type(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"  {key}: type={type(value)} shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")
        weight = obj.get("weight", obj.get("sparse_linear.weight"))
        bias = obj.get("bias", obj.get("sparse_linear.bias"))
    else:
        print("  tensor shape=", getattr(obj, "shape", None), "dtype=", getattr(obj, "dtype", None))
        weight, bias = obj, None
    if weight is None:
        raise RuntimeError("sparse_linear.pt 里没有找到 weight")
    # 稀疏权重必须在 fp32 下算，否则 fp16 会下溢成 0
    weight = weight.detach().float()
    if weight.numel() != 1024:
        raise RuntimeError(f"weight 元素数不是 1024：{weight.shape}")
    weight = weight.reshape(1, 1024)
    bias = None if bias is None else bias.detach().float().reshape(-1)[:1]
    print("  weight 归一化后 shape:", tuple(weight.shape), "bias:", None if bias is None else tuple(bias.shape))
    return weight, bias


def load_model():
    """加载 tokenizer 和模型到可用设备，返回 (tokenizer, model, device)。"""
    print("\n=== 2. 加载 BGE-M3 ===")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    # 模型目录只有 pytorch_model.bin，必须显式 use_safetensors=False
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        use_safetensors=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    print(f"device={device} dtype={next(model.parameters()).dtype} load={time.perf_counter() - t0:.1f}s")
    if device == "cuda":
        print(f"显存占用: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")
    return tokenizer, model, device


@torch.inference_mode()
def encode(tokenizer, model, device, weight, bias):
    """对 TEXTS 做一次前向，返回 (dense 张量, sparse 字典列表)，并打印观测信息。"""
    print("\n=== 3. dense + sparse 编码 ===")
    t0 = time.perf_counter()
    batch = tokenizer(
        TEXTS,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    hidden = model(**batch).last_hidden_state
    # dense：CLS 池化 + L2 归一化
    dense = F.normalize(hidden[:, 0].float(), p=2, dim=-1).cpu()

    # sparse：线性层投影 + ReLU；bias 为 None 时加 0 简化分支
    token_weights = torch.relu(hidden.float() @ weight.to(device).T + (bias.to(device) if bias is not None else 0.0))
    token_weights = token_weights.squeeze(-1)

    mask = batch["attention_mask"].bool()
    input_ids = batch["input_ids"]
    special_ids = set(tokenizer.all_special_ids)
    sparse_vectors = []
    for row in range(len(TEXTS)):
        # 跳过 padding 和特殊 token，同一 token 取最大权重
        ids = input_ids[row][mask[row]].tolist()
        weights = token_weights[row][mask[row]].tolist()
        aggregated: dict[int, float] = {}
        for token_id, token_weight in zip(ids, weights):
            if token_id in special_ids or token_weight <= 1e-6:
                continue
            aggregated[token_id] = max(aggregated.get(token_id, 0.0), float(token_weight))
        # Milvus 不接受空稀疏向量
        if not aggregated:
            aggregated = {0: 1e-7}
        sparse_vectors.append(aggregated)

    print(f"编码耗时: {time.perf_counter() - t0:.2f}s, dense shape={tuple(dense.shape)}")
    print("dense 范数:", [round(float(x.norm()), 6) for x in dense])
    for i, sparse in enumerate(sparse_vectors):
        # 解码 token id，确认稀疏权重落在实词上
        top = sorted(sparse.items(), key=lambda item: item[1], reverse=True)[:6]
        words = [(tokenizer.decode([token_id]), round(score, 3)) for token_id, score in top]
        print(f"  稀疏向量[{i}] 长度={len(sparse)} top={words}")
    if device == "cuda":
        print(f"编码后显存: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")
    return dense, sparse_vectors


def probe_milvus(dense, sparse_vectors):
    """在临时目录建一个最小集合，验证 milvus-lite 的双路检索能力。"""
    print("\n=== 4. milvus-lite dense + sparse 检索 ===")
    from pymilvus import DataType, MilvusClient

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_probe_"))
    uri = str(tmp_dir / "probe.db")
    client = MilvusClient(uri=uri)
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=512)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="FLAT", metric_type="IP")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        client.create_collection("probe", schema=schema, index_params=index_params)

        rows = [
            {
                "pk": f"p{i}",
                "text": text,
                "dense": dense[i].tolist(),
                "sparse": sparse_vectors[i],
            }
            for i, text in enumerate(TEXTS)
        ]
        client.insert(collection_name="probe", data=rows)
        client.load_collection(collection_name="probe")
        print("行数:", client.query(collection_name="probe", filter="pk != ''", output_fields=["count(*)"]))
        # 用第一条文本的向量检索，第一条应排第一（自匹配）
        dense_hits = client.search(
            collection_name="probe",
            data=[dense[0].tolist()],
            anns_field="dense",
            limit=3,
            output_fields=["text"],
            search_params={"metric_type": "IP", "params": {}},
        )
        print("dense top3:", [(hit["entity"]["text"], round(hit["distance"], 4)) for hit in dense_hits[0]])

        sparse_hits = client.search(
            collection_name="probe",
            data=[sparse_vectors[0]],
            anns_field="sparse",
            limit=3,
            output_fields=["text"],
            search_params={"metric_type": "IP", "params": {}},
        )
        print("sparse top3:", [(hit["entity"]["text"], round(hit["distance"], 4)) for hit in sparse_hits[0]])
        print("milvus-lite 稀疏检索: OK")
    finally:
        # 清理临时库，drop 失败也不影响结论，只提示一下
        try:
            client.drop_collection(collection_name="probe")
        except Exception as exc:  # noqa: BLE001
            print("drop_collection 失败（可忽略）:", exc)
        client.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    """按顺序跑通三个高风险点：权重结构 -> 模型加载 -> 编码 -> 向量库检索。"""
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    weight, bias = inspect_sparse_linear()
    tokenizer, model, device = load_model()
    dense, sparse_vectors = encode(tokenizer, model, device, weight, bias)
    probe_milvus(dense, sparse_vectors)
    print("\nM0 全部通过")


if __name__ == "__main__":
    main()
