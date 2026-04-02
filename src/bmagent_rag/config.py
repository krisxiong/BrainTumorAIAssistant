# 本文件负责读取 .env 和组装后端建库所需的 SyncConfig 配置对象。
# 同步脚本、qa_service.py 和 qa_api.py 都会依赖这里的配置装配逻辑。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable


DEFAULT_ALLOWED_EXTENSIONS = (
    ".pdf",
    ".txt",
    ".md",
    ".docx",

    ".html",
    ".htm",
    ".csv",
    ".json",
)


@dataclass(slots=True)
# SyncConfig 类型或服务类，服务于本地 RAG 配置装配。
class SyncConfig:
    source_dir: Path
    state_dir: Path
    manifest_path: Path
    index_path: Path
    knowledge_base_name: str
    allowed_extensions: tuple[str, ...]
    chunk_size_chars: int
    chunk_overlap_chars: int
    dry_run: bool = False
    knowledge_base_id: str | None = None

    @property
    # 处理 vector_store_name 相关逻辑，供本地 RAG 配置装配使用。
    def vector_store_name(self) -> str:
        return self.knowledge_base_name

    @vector_store_name.setter
    # 处理 vector_store_name 相关逻辑，供本地 RAG 配置装配使用。
    def vector_store_name(self, value: str) -> None:
        self.knowledge_base_name = value

    @property
    # 处理 vector_store_id 相关逻辑，供本地 RAG 配置装配使用。
    def vector_store_id(self) -> str | None:
        return self.knowledge_base_id

    @vector_store_id.setter
    # 处理 vector_store_id 相关逻辑，供本地 RAG 配置装配使用。
    def vector_store_id(self, value: str | None) -> None:
        self.knowledge_base_id = value


# 读取 .env 文件中的 KEY=VALUE 配置，仅在当前进程没有同名变量时才写入。
def load_env_file(path: Path) -> None:

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# 处理 _project_root 相关逻辑，供本地 RAG 配置装配使用。
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 处理 _read_path 相关逻辑，供本地 RAG 配置装配使用。
def _read_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


# 处理 _read_int 相关逻辑，供本地 RAG 配置装配使用。
def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


# 处理 _read_extensions 相关逻辑，供本地 RAG 配置装配使用。
def _read_extensions(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_ALLOWED_EXTENSIONS
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    normalized = tuple(value if value.startswith(".") else f".{value}" for value in values)
    return normalized or DEFAULT_ALLOWED_EXTENSIONS


# 构建 SyncConfig 对象，统一后端建库和状态接口对目录和参数的理解。
def build_config(
    source_dir: Path | None = None,
    state_dir: Path | None = None,
    manifest_path: Path | None = None,
    index_path: Path | None = None,
    knowledge_base_name: str | None = None,
    vector_store_name: str | None = None,
    knowledge_base_id: str | None = None,
    vector_store_id: str | None = None,
    allowed_extensions: Iterable[str] | None = None,
    chunk_size_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
    dry_run: bool = False,
) -> SyncConfig:
    repo_root = _project_root()
    resolved_source = source_dir or _read_path(
        "BMAGENT_KB_SOURCE_DIR", str(repo_root / "data" / "knowledge_base" / "source")
    )
    resolved_state = state_dir or _read_path(
        "BMAGENT_KB_STATE_DIR", str(repo_root / "data" / "knowledge_base" / "state")
    )
    resolved_manifest = manifest_path or resolved_state / "manifest.json"
    resolved_index = index_path or _read_path(
        "BMAGENT_KB_INDEX_PATH", str(resolved_state / "local_index.json")
    )
    resolved_name = knowledge_base_name or vector_store_name or os.getenv(
        "BMAGENT_KB_NAME", "brain-tumor-mri-kb-local"
    )
    resolved_extensions = (
        tuple(ext.lower() for ext in allowed_extensions)
        if allowed_extensions
        else _read_extensions(os.getenv("BMAGENT_KB_ALLOWED_EXTENSIONS"))
    )
    resolved_chunk_size = (
        chunk_size_chars
        if chunk_size_chars is not None
        else _read_int("BMAGENT_KB_CHUNK_SIZE_CHARS", 1400)
    )
    resolved_chunk_overlap = (
        chunk_overlap_chars
        if chunk_overlap_chars is not None
        else _read_int("BMAGENT_KB_CHUNK_OVERLAP_CHARS", 250)
    )

    return SyncConfig(
        source_dir=resolved_source,
        state_dir=resolved_state,
        manifest_path=resolved_manifest,
        index_path=resolved_index,
        knowledge_base_name=resolved_name,
        allowed_extensions=tuple(resolved_extensions),
        chunk_size_chars=resolved_chunk_size,
        chunk_overlap_chars=resolved_chunk_overlap,
        dry_run=dry_run,
        knowledge_base_id=knowledge_base_id or vector_store_id or os.getenv("BMAGENT_KB_ID") or None,
    )
