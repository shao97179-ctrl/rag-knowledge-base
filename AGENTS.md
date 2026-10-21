# AGENTS.md

## 项目性质

企业知识库 RAG 学习/简历项目，重点展示 BGE-M3 混合检索、父子块、权限隔离和可量化评估。不是生产可用的知识库系统。

技术栈：Python 3.12 + LangChain（splitter / ChatOpenAI / prompt）+ 裸 transformers + pymilvus / milvus-lite + SQLite + FastAPI。

## 环境

- Python 3.12，虚拟环境 `.venv`，直接使用 `.venv/bin/python`。
- BGE-M3 本地模型：默认从项目根下的 `BGE-M3/` 读取，也可用 `RAG_BGE_M3_PATH` 指定绝对路径（fp16，约 2.2GB）。
- `pyproject.toml` 是依赖来源。不要擅自新增依赖。
- `.env` 可能包含 API Key，**不要读取、打印或提交**；`.gitignore` 已忽略。
- 所有配置键使用 `RAG_` 前缀，路径相对项目根解析。

## 常用命令

```bash
.venv/bin/python -m compileall src
.venv/bin/pytest -q
.venv/bin/python -m rag_learning probe
.venv/bin/python -m rag_learning ingest
.venv/bin/python -m rag_learning ask -q "UPI 的单笔和每日交易限额是多少" --dept ops --clearance internal
.venv/bin/python -m rag_learning eval --no-faithfulness
.venv/bin/uvicorn rag_learning.api:app --host 127.0.0.1 --port 8000
```

真实 LLM 路径（`ask` 有 Key 时、`eval` 不带 `--no-faithfulness`、`probe --live`）会产生费用；测试不调用 LLM。

## 架构边界

- `config.py`：只做配置和路径解析，不写业务。
- `loaders/`：只负责把文件解析成文本 + 权限元数据，不做切分。
- `chunker.py`：表格原子单元必须在父块切分**之前**提取；表格不参与长度切分。
- `embedder.py`：只负责 dense/sparse 编码；不要把检索逻辑塞进来。
- `vector_store.py`：所有 Milvus schema、upsert、delete、search 都在这里；查询前必须 `load_collection`。
- `registry.py`：所有 SQLite SQL 都在这里。
- `ingest.py`：编排解析、切分、编码、写入；增量判断和尾巴删除只在这里。
- `retrieval/`：查询改写、dense/sparse 双路、RRF、ACL filter。
- `generation/`：Prompt、引用、拒答、离线降级。
- `pipeline.py`：端到端组装，API / CLI / 评估共用。
- `api.py`：只做 HTTP 入参/出参和状态码，不写业务 SQL。
- `cli.py`：Typer 入口；`main` 由 `__init__.py` 导出。
- `tests/`：必须离线可跑，不加载 BGE-M3、不调用 LLM；用 FakeEmbedder 和桩。

## 数据与索引规则

- 主键必须确定性：`sha1(doc_id#chunk_index)`，不要用 auto_id。
- 文档变短时先 `delete_tail` 再 `upsert`，顺序不能反。
- `--force` 重建后要验证行数；milvus-lite drop 后可能残留，必要时物理删库文件。
- ACL filter 必须在 Milvus 检索参数里执行，生成前再做一次 Python 校验兜底。
- `parent_text` 冗余存在每个子块行里，换取检索后少一次查询；这是刻意的取舍。
- 稀疏向量不能是空 dict，空时用 `{0: 1e-7}`。
- Milvus VARCHAR `max_length` 是字节数，不是字符数。

## 语料规则

- `zh_src/` 是中文 Markdown 母版，命名 `<部门>__<密级>__<slug>.md`；`docs/` 由它生成。
- `docs/` 必须是 `docs/<部门>/<密级>/` 两层。加中间层（如 `docs/zh-bank/ops/...`）会让 `infer_acl` 退化成 `general/public`。
- **每篇只部署一种格式**（按 pdf / md / docx / html / pptx 轮换）。同一内容部署多份格式会被索引多次，导致检索结果重复、评估匹配错文档。
- 格式轮换会让 `eval/golden_set.jsonl` 里的扩展名失效：重新部署后必须按 stem 重新解析实际扩展名。
- 生成与校验统一走 `scripts/build_zh_corpus.py`，不要手写转换脚本。
- 语料翻译自 udayallu/RAG-Multi-Corpus（MIT），版权声明见 `NOTICE`。


## 验证要求

每次改动至少运行：

```bash
.venv/bin/python -c "from rag_learning import main"
.venv/bin/pytest -q
.venv/bin/python -m rag_learning probe
```

涉及入库/检索改动时再运行 `ingest` 两次（第二次应全部 unchanged）和 `eval --no-faithfulness`。不要虚构测试结果，必须贴真实输出。

## 已知坑

- milvus-lite 跨进程查询会处于 released 状态，必须先 `load_collection`。
- transformers 5.3 加载 `pytorch_model.bin` 需要 `use_safetensors=False`。
- `sparse_linear.pt` 是 OrderedDict（fp16 weight + bias），不是裸 tensor。
- pymilvus 的 hit 主键字段名是 `pk`，不是 `id`。
- 不要一边 `serve` 一边 `ingest`；milvus-lite 是单机嵌入式。
- 本机 LibreOffice 在沙箱里无法加载文件；演示 PDF 用 reportlab + STSong-Light。

## 风格

- 代码标识符用英文，注释和面向用户的文案用中文。
- 最小 diff，不要顺手重构无关文件。
- 新增 API 时同步更新 README 的 API 表和 `tests/test_api.py`。
