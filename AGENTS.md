# AGENTS.md

企业知识库 RAG 学习/简历项目，重点是 BGE-M3 混合检索、父子块、权限隔离和可量化评估。
技术栈：Python 3.12 + LangChain（splitter / ChatOpenAI / prompt）+ 裸 transformers + pymilvus / milvus-lite + SQLite + FastAPI。

## 环境

- 虚拟环境在 `.venv`，统一用 `.venv/bin/python` 跑。
- BGE-M3 模型默认放项目根下的 `BGE-M3/`（fp16，约 2.2GB），也可以用 `RAG_BGE_M3_PATH` 指定绝对路径。
- 依赖以 `pyproject.toml` 为准，不要私自加包。
- `.env` 里有 API Key，不要读取、打印或提交，`.gitignore` 已忽略。
- 配置键统一 `RAG_` 前缀，路径相对项目根解析。

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

`ask` 带 Key、`eval` 不带 `--no-faithfulness`、`probe --live` 都会真实调 LLM 产生费用；测试不调 LLM。

## 架构边界

- `config.py` 只管配置和路径解析。
- `loaders/` 只把文件解析成文本 + 权限元数据，不做切分。
- `chunker.py` 表格原子单元要在父块切分之前提取，表格不参与长度切分。
- `embedder.py` 只做 dense/sparse 编码，检索逻辑别放这里。
- `vector_store.py` 收拢所有 Milvus 的 schema、upsert、delete、search，查询前先 `load_collection`。
- `registry.py` 收拢所有 SQLite SQL。
- `ingest.py` 负责编排解析、切分、编码、写入；增量判断和尾巴删除只在这里做。
- `retrieval/` 查询改写、dense/sparse 双路、RRF、ACL filter。
- `generation/` Prompt、引用、拒答、离线降级。
- `pipeline.py` 端到端组装，API / CLI / 评估共用。
- `api.py` 只做 HTTP 入参/出参和状态码，不写业务 SQL。
- `cli.py` 是 Typer 入口，`main` 由 `__init__.py` 导出。
- `tests/` 必须离线可跑：不加载 BGE-M3、不调 LLM，用 FakeEmbedder 和桩。

## 数据与索引

- 主键是 `sha1(doc_id#chunk_index)`，不要换成 auto_id。
- 文档变短时先 `delete_tail` 再 `upsert`，顺序不能反。
- `--force` 重建后要验证行数；milvus-lite drop 后可能残留，必要时直接删库文件。
- ACL filter 在 Milvus 检索参数里执行，生成前再用 Python 校验一次兜底。
- `parent_text` 冗余存进每个子块行，换检索后少一次查询。
- 稀疏向量不能是空 dict，空时填 `{0: 1e-7}`。
- Milvus VARCHAR 的 `max_length` 是字节数，中文一个字 3 字节。

## 语料

- `zh_src/` 是中文 Markdown 母版，命名 `<部门>__<密级>__<slug>.md`，`docs/` 由它生成。
- `docs/` 必须是 `docs/<部门>/<密级>/` 两层；中间多一层（比如 `docs/zh-bank/ops/...`）会让 `infer_acl` 退化成 `general/public`。
- 每篇内容只部署一种格式（pdf / md / docx / html / pptx 轮换）；同一内容多格式会重复索引，检索出重复结果，评估也会匹配错文档。
- 重新部署后 `eval/golden_set.jsonl` 里的扩展名会失效，要按 stem 重解析实际扩展名。
- 转换和校验统一走 `scripts/build_zh_corpus.py`，不要手写转换脚本。
- 语料翻译自 udayallu/RAG-Multi-Corpus（MIT），版权见 `NOTICE`。

## 验证

每次改动至少跑：

```bash
.venv/bin/python -c "from rag_learning import main"
.venv/bin/pytest -q
.venv/bin/python -m rag_learning probe
```

动了入库或检索的话，再加跑两次 `ingest`（第二次应全部 unchanged）和 `eval --no-faithfulness`。贴真实输出，不要编造。

## 已知坑

- milvus-lite 跨进程查询会处于 released 状态，先 `load_collection`。
- transformers 5.3 加载 `pytorch_model.bin` 要 `use_safetensors=False`。
- `sparse_linear.pt` 是 OrderedDict（fp16 weight + bias），不是裸 tensor。
- pymilvus 的 hit 主键字段名是 `pk`，不是 `id`。
- 不要一边 `serve` 一边 `ingest`，milvus-lite 是单机嵌入式。
- 本机 LibreOffice 在沙箱里加载不了文件，演示 PDF 用 reportlab + STSong-Light 生成。

## 风格

- 标识符用英文，注释和用户可见的文案用中文。
- 改动保持最小 diff，别顺手重构无关文件。
- 新增 API 时同步更新 README 的 API 表和 `tests/test_api.py`。
