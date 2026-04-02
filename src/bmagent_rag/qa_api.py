# 本文件对外暴露 FastAPI 接口，连接前端、qa_service.py 和本地知识库状态服务。
# 你读入口 API 和调试接口行为时，可以先看这个文件。

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from .config import build_config, load_env_file

from .manifest import KnowledgeBaseManifest
from .qa_config import build_qa_config
from .qa_models import (
    BrainTumorQaEnvelope,
    BrainTumorQaRequest,
    KnowledgeBaseDocumentStatus,
    KnowledgeBaseStatus,
    KnowledgeBaseSyncRequest,
    KnowledgeBaseSyncResponse,
    QaSessionState,
    SessionCreateRequest,
)
from .qa_service import BrainTumorQaService
from .sync import sync_knowledge_base


load_env_file(Path('.env'))
qa_config = build_qa_config()
qa_service = BrainTumorQaService(config=qa_config)


app = FastAPI(title='Brain Tumor MRI Assistant', version='0.2.0')


@app.get('/healthz')
# 处理 healthz 相关逻辑，供FastAPI 接口层使用。
def healthz() -> dict[str, str | None]:
    return {
        'status': 'ok',
        'model': qa_config.openai_model,
        'base_url': qa_config.openai_base_url,
        'retrieval_mode': 'local_rag',
    }


@app.get('/api/healthz')
# 处理 api_healthz 相关逻辑，供FastAPI 接口层使用。
def api_healthz() -> dict[str, str | None]:
    return healthz()


@app.post('/api/sessions', response_model=QaSessionState)
# 处理 create_session 相关逻辑，供FastAPI 接口层使用。
def create_session(payload: SessionCreateRequest) -> QaSessionState:
    return qa_service.create_session(session_id=payload.session_id, title=payload.title)


@app.get('/api/sessions/{session_id}', response_model=QaSessionState)
# 处理 get_session 相关逻辑，供FastAPI 接口层使用。
def get_session(session_id: str) -> QaSessionState:
    try:
        return qa_service.load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='session not found') from exc


@app.post('/api/qa', response_model=BrainTumorQaEnvelope)
# 问答主接口，将请求交由 BrainTumorQaService.answer() 处理。
def qa(request: BrainTumorQaRequest) -> BrainTumorQaEnvelope:
    return qa_service.answer(request)


@app.post('/qa', response_model=BrainTumorQaEnvelope, include_in_schema=False)
# 处理 qa_legacy 相关逻辑，供FastAPI 接口层使用。
def qa_legacy(request: BrainTumorQaRequest) -> BrainTumorQaEnvelope:
    return qa(request)


@app.get('/api/kb/status', response_model=KnowledgeBaseStatus)
# 返回本地知识库的索引状态、文档列表和统计信息。
def knowledge_base_status() -> KnowledgeBaseStatus:
    sync_config = build_config()
    manifest = KnowledgeBaseManifest.load(sync_config.manifest_path)
    source_root = manifest.source_root or str(sync_config.source_dir)

    documents = [
        KnowledgeBaseDocumentStatus(
            relative_path=record.relative_path,
            absolute_path=record.absolute_path,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            mime_type=record.mime_type,
            parser_name=record.parser_name,
            extracted_char_count=record.extracted_char_count,
            chunk_count=record.chunk_count,
            last_indexed_at=record.last_indexed_at,
            last_synced_at=record.last_synced_at,
            sync_state=record.sync_state,
            error=record.error,
        )
        for record in sorted(manifest.documents.values(), key=lambda item: item.relative_path)
    ]
    indexed_count = sum(1 for item in documents if item.sync_state == 'indexed')
    failed_count = sum(1 for item in documents if item.sync_state == 'failed')
    pending_count = max(len(documents) - indexed_count - failed_count, 0)

    return KnowledgeBaseStatus(
        source_root=source_root,
        manifest_path=str(sync_config.manifest_path),
        index_path=str(sync_config.index_path),
        knowledge_base_name=manifest.knowledge_base_name or sync_config.knowledge_base_name,
        knowledge_base_id=manifest.knowledge_base_id,
        source_dir_exists=sync_config.source_dir.exists(),
        state_dir_exists=sync_config.state_dir.exists(),
        total_documents=len(documents),
        indexed_documents=indexed_count,
        pending_documents=pending_count,
        failed_documents=failed_count,
        chunk_count=manifest.chunk_count,
        chunk_size_chars=manifest.chunk_size_chars,
        chunk_overlap_chars=manifest.chunk_overlap_chars,
        last_sync_at=manifest.last_sync_at,
        updated_at=manifest.updated_at,
        documents=documents,
    )


@app.post('/api/kb/sync', response_model=KnowledgeBaseSyncResponse)
# 触发本地建库或同步流程，并返回执行摘要。
def knowledge_base_sync(payload: KnowledgeBaseSyncRequest) -> KnowledgeBaseSyncResponse:
    sync_config = build_config(
        source_dir=Path(payload.source_dir) if payload.source_dir else None,
        state_dir=Path(payload.state_dir) if payload.state_dir else None,
        manifest_path=Path(payload.manifest_path) if payload.manifest_path else None,
        index_path=Path(payload.index_path) if payload.index_path else None,
        knowledge_base_name=payload.knowledge_base_name,
        knowledge_base_id=payload.knowledge_base_id,
        chunk_size_chars=payload.chunk_size_chars,
        chunk_overlap_chars=payload.chunk_overlap_chars,
        dry_run=payload.dry_run,
    )
    summary = sync_knowledge_base(sync_config)
    return KnowledgeBaseSyncResponse(
        source_root=str(summary.source_root),
        manifest_path=str(summary.manifest_path),
        index_path=str(summary.index_path),
        knowledge_base_id=summary.knowledge_base_id,
        total_files=summary.total_files,
        new_or_changed_files=summary.new_or_changed_files,
        skipped_files=summary.skipped_files,
        indexed_files=summary.indexed_files,
        failed_files=summary.failed_files,
        chunk_count=summary.chunk_count,
        dry_run=summary.dry_run,
    )
