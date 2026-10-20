"""项目配置：所有环境变量统一使用 RAG_ 前缀，路径相对项目根解析。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：本文件在 src/rag_learning/ 下，向上三层即仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全局配置对象，从环境变量 / .env 读取，字段名自动加 RAG_ 前缀。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # ---- 目录与向量库 ----
    docs_dir: Path = Path("docs")
    data_dir: Path = Path("data")
    milvus_uri: str = "data/milvus.db"
    milvus_collection: str = "rag_chunks"

    # ---- BGE-M3 编码参数（显存有限，默认 fp16 + batch 8 + 1024 长度）----
    # 默认值只是占位：把模型目录放到项目根下的 BGE-M3/，或用 RAG_BGE_M3_PATH 指定绝对路径
    bge_m3_path: Path = Path("BGE-M3")
    embed_device: str = "auto"
    embed_dtype: str = "float16"
    embed_batch_size: int = 8
    embed_max_length: int = 1024
    query_max_length: int = 256

    # ---- 父子块切分参数 ----
    parent_size: int = 1200
    child_size: int = 350
    chunk_overlap: int = 50

    # ---- 检索参数：两路各召回 30，融合后保留 6 个父块 ----
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_final: int = 6
    rrf_k: int = 60
    max_context_chars: int = 6000

    # ---- LLM 参数 ----
    llm_provider: str = "zhipu"
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-4.6v"
    llm_light_model: str = "glm-4.6v"
    llm_temperature: float = 0.1
    llm_timeout: int = 60

    # API Key 兼容带前缀和不带前缀两种环境变量写法。
    # 注意：RAG_ 前缀是项目约定，但很多用户习惯直接写 ZHIPUAI_API_KEY，
    # 所以这里用 AliasChoices 同时接受两种名字。
    zhipuai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ZHIPUAI_API_KEY", "RAG_ZHIPUAI_API_KEY"),
    )
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "RAG_DEEPSEEK_API_KEY"),
    )

    log_level: str = "INFO"

    @field_validator("embed_device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        """校验推理设备只能是 auto / cuda / cpu，并统一为小写。"""
        value = value.strip().lower()
        if value not in {"auto", "cuda", "cpu"}:
            raise ValueError("RAG_EMBED_DEVICE 只能是 auto / cuda / cpu")
        return value

    @field_validator("embed_dtype")
    @classmethod
    def _validate_dtype(cls, value: str) -> str:
        """校验权重精度只能是 float16 / bfloat16 / float32。"""
        value = value.strip().lower()
        if value not in {"float16", "bfloat16", "float32"}:
            raise ValueError("RAG_EMBED_DTYPE 只能是 float16 / bfloat16 / float32")
        return value

    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """校验 LLM 供应商只能是 zhipu / deepseek / offline。"""
        value = value.strip().lower()
        if value not in {"zhipu", "deepseek", "offline"}:
            raise ValueError("RAG_LLM_PROVIDER 只能是 zhipu / deepseek / offline")
        return value

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        """把所有相对路径解析成绝对路径，并在切换 provider 时纠正默认模型名。"""
        # 相对路径统一相对项目根解析，避免受启动时 cwd 影响
        if not self.docs_dir.is_absolute():
            self.docs_dir = (PROJECT_ROOT / self.docs_dir).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        if not self.bge_m3_path.is_absolute():
            self.bge_m3_path = (PROJECT_ROOT / self.bge_m3_path).resolve()
        # milvus_uri 既可能是本地文件路径，也可能是 http(s) 服务地址
        if not (self.milvus_uri.startswith("http://") or self.milvus_uri.startswith("https://")):
            uri_path = Path(self.milvus_uri)
            if not uri_path.is_absolute():
                uri_path = (PROJECT_ROOT / uri_path).resolve()
            self.milvus_uri = str(uri_path)

        # 切换 provider 时自动纠正另一家的默认 base_url / 模型名
        if self.llm_provider == "deepseek":
            if "bigmodel" in self.llm_base_url:
                self.llm_base_url = "https://api.deepseek.com"
            if self.llm_model.startswith("glm"):
                self.llm_model = "deepseek-chat"
            if self.llm_light_model.startswith("glm"):
                self.llm_light_model = "deepseek-chat"
        elif self.llm_provider == "zhipu":
            if "deepseek" in self.llm_base_url:
                self.llm_base_url = "https://open.bigmodel.cn/api/paas/v4"
            if self.llm_model.startswith("deepseek"):
                self.llm_model = "glm-4.6v"
            if self.llm_light_model.startswith("deepseek"):
                self.llm_light_model = "glm-4.6v"
        return self

    @property
    def api_key(self) -> str | None:
        """按当前 provider 取对应的 Key；offline 或空字符串都视为没有 Key。"""
        if self.llm_provider == "zhipu":
            key = self.zhipuai_api_key
        elif self.llm_provider == "deepseek":
            key = self.deepseek_api_key
        else:
            return None
        key = (key or "").strip()
        return key or None

    @property
    def llm_available(self) -> bool:
        """是否具备真实调用 LLM 的条件：非 offline 且配置了 Key。"""
        return self.llm_provider != "offline" and self.api_key is not None

    @property
    def web_dir(self) -> Path:
        """单页 Web UI 所在目录，跟着代码走而不是跟着配置走。"""
        return Path(__file__).resolve().parent / "web"

    @property
    def registry_path(self) -> Path:
        """SQLite 文档注册表路径，固定放在 data 目录下。"""
        return self.data_dir / "registry.sqlite"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内只读一次 .env；测试里可用 get_settings.cache_clear() 重置。"""
    return Settings()
