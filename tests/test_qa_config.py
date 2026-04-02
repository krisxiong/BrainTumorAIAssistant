from __future__ import annotations

from pathlib import Path

from bmagent_rag.qa_config import build_qa_config


def test_build_qa_config_reads_openai_base_url(tmp_path: Path) -> None:
    env_file = tmp_path / '.env'
    env_file.write_text(
        'OPENAI_API_KEY=test-key\nOPENAI_BASE_URL=https://example.com/v1\n',
        encoding='utf-8',
    )

    config = build_qa_config(env_file=env_file)

    assert config.openai_api_key == 'test-key'
    assert config.openai_base_url == 'https://example.com/v1'