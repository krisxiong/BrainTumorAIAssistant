"""本模块负责生成模型侧配置的解析。

项目当前采用“本地检索 + 远端生成”的架构，因此需要把检索配置和
模型调用配置拆开管理。qa_api.py 在启动时会加载这里的配置，
qa_service.py 在真正调用 Yunwu 或其他 OpenAI-compatible 接口时使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from .config import load_env_file


@dataclass(slots=True)
class QaConfig:
    """远端生成模型配置对象。

    它记录 API 密钥、Base URL、模型名、推理强度、输出长度等参数，
    供问答服务统一使用。
    """

    openai_api_key: str
    openai_model: str = 'gpt-5.4'
    openai_base_url: str | None = None
    openai_vector_store_id: str | None = None
    openai_reasoning_effort: str = 'medium'
    openai_text_verbosity: str = 'low'
    openai_max_num_results: int = 5
    openai_max_output_tokens: int = 1400


def _read_env_file_values(path: Path) -> dict[str, str]:
    """读取显式传入的 .env 文件内容。

    这个辅助函数不修改全局环境变量，而是返回一个局部字典，供
    build_qa_config() 在“显式 env_file 参数优先”这一语义下使用。
    """

    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip().lstrip('\ufeff')
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def build_qa_config(env_file: Path | None = None) -> QaConfig:
    """从环境变量和可选 .env 文件构造 QaConfig。

    正常应用启动时，qa_api.py 会先调用 load_env_file(Path('.env'))，然后
    再调用这里的 build_qa_config()。测试或脚本场景下如果显式传入 env_file，
    则该文件中的值应优先于当前进程中已存在的环境变量，以便进行隔离测试。
    """

    file_values: dict[str, str] = {}
    if env_file is not None:
        load_env_file(env_file)
        file_values = _read_env_file_values(env_file)

    def read(name: str, default: str = '') -> str:
        return file_values.get(name, os.getenv(name, default))

    return QaConfig(
        openai_api_key=read('OPENAI_API_KEY', ''),
        openai_model=read('OPENAI_MODEL', 'gpt-5.4'),
        openai_base_url=read('OPENAI_BASE_URL', '') or None,
        openai_vector_store_id=read('OPENAI_VECTOR_STORE_ID', '') or None,
        openai_reasoning_effort=read('OPENAI_REASONING_EFFORT', 'medium'),
        openai_text_verbosity=read('OPENAI_TEXT_VERBOSITY', 'low'),
        openai_max_num_results=int(read('OPENAI_MAX_NUM_RESULTS', '5')),
        openai_max_output_tokens=int(read('OPENAI_MAX_OUTPUT_TOKENS', '1400')),
    )