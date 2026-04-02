# ?????? source ????????????????? local_rag.py ???????
# manifest ????????????????/api/kb/status ????????
from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import SyncConfig
from .local_rag import LocalDocument, build_knowledge_base_id, build_local_index, save_local_index
from .manifest import DocumentRecord, KnowledgeBaseManifest

PARSER_ERROR_MESSAGES = {
    'pdf_missing_dependency': '?????????? pypdf?PDF ?????????????????????',
    'docx_missing_dependency': '?????????? python-docx?DOCX ?????????????????????',
    'pdf_parse_failed': 'PDF ???????????????????',
    'docx_parse_failed': 'DOCX ???????????????????',
    'pdf_filtered_as_noise': 'PDF ??????????????????????????',
    'docx_filtered_as_noise': 'DOCX ?????????????????????',
}


# ????????????????CLI ?API ????????
@dataclass(slots=True)
class SyncSummary:
    source_root: Path
    manifest_path: Path
    index_path: Path
    knowledge_base_id: str
    total_files: int
    new_or_changed_files: int
    skipped_files: int
    indexed_files: int
    failed_files: int
    chunk_count: int
    dry_run: bool

    @property
    # ?? vector_store_id ????????????????
    def vector_store_id(self) -> str:
        return self.knowledge_base_id


# ????????????????????????
@dataclass(slots=True)
class LocalSearchSummary:
    relative_path: str
    score: float
    snippet: str


# ???? SHA256???????????
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ???????? MIME ???
def infer_mime_type(path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type:
        return mime_type
    if path.suffix.lower() in {'.md', '.txt', '.csv', '.json'}:
        return 'text/plain'
    return None


# ?????????????????? LocalDocument ???
def scan_documents(source_root: Path, allowed_extensions: Sequence[str]) -> list[LocalDocument]:
    allowed = {ext.lower() for ext in allowed_extensions}
    documents: list[LocalDocument] = []
    for path in sorted(source_root.rglob('*')):
        if not path.is_file() or path.name.startswith('.'):
            continue
        if path.suffix.lower() not in allowed:
            continue
        documents.append(
            LocalDocument(
                absolute_path=path,
                relative_path=path.relative_to(source_root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                mime_type=infer_mime_type(path),
            )
        )
    return documents


# ?????????? ID?
def ensure_knowledge_base_id(manifest: KnowledgeBaseManifest, config: SyncConfig) -> str:
    if config.knowledge_base_id:
        manifest.knowledge_base_id = config.knowledge_base_id
        manifest.vector_store_id = config.knowledge_base_id
        manifest.knowledge_base_name = config.knowledge_base_name
        manifest.vector_store_name = config.knowledge_base_name
        return config.knowledge_base_id
    if manifest.knowledge_base_id:
        manifest.vector_store_id = manifest.knowledge_base_id
        return manifest.knowledge_base_id
    if manifest.vector_store_id:
        manifest.knowledge_base_id = manifest.vector_store_id
        return manifest.vector_store_id

    knowledge_base_id = build_knowledge_base_id(config.knowledge_base_name, config.source_dir)
    manifest.knowledge_base_id = knowledge_base_id
    manifest.vector_store_id = knowledge_base_id
    manifest.knowledge_base_name = config.knowledge_base_name
    manifest.vector_store_name = config.knowledge_base_name
    return knowledge_base_id


# ????????
def ensure_vector_store_id(manifest: KnowledgeBaseManifest, config: SyncConfig) -> str:
    return ensure_knowledge_base_id(manifest, config)


# ????????? manifest ???????
def _infer_record_error(parser_name: str | None, chunk_count: int) -> str | None:
    if parser_name in PARSER_ERROR_MESSAGES:
        return PARSER_ERROR_MESSAGES[parser_name]
    if chunk_count <= 0 and parser_name in {'pdf', 'docx'}:
        return '??????????????????????'
    return None


# ????????? manifest??? API ?????????????
def _build_manifest_records(
    manifest: KnowledgeBaseManifest,
    documents: list[LocalDocument],
    indexed_at: datetime,
    parser_summaries: dict[str, object],
) -> None:
    manifest.documents = {}
    for document in documents:
        summary = parser_summaries.get(document.relative_path)
        parser_name = getattr(summary, 'parser_name', None)
        extracted_char_count = int(getattr(summary, 'extracted_char_count', 0))
        chunk_count = int(getattr(summary, 'chunk_count', 0))
        error = _infer_record_error(parser_name, chunk_count)
        sync_state = 'failed' if error else 'indexed'
        record = DocumentRecord(
            relative_path=document.relative_path,
            absolute_path=str(document.absolute_path),
            size_bytes=document.size_bytes,
            sha256=document.sha256,
            mime_type=document.mime_type,
            openai_file_id=f'local://{document.sha256[:16]}',
            openai_file_status='local-indexed' if not error else 'local-failed',
            vector_store_attachment_id=f"{manifest.knowledge_base_id or 'local'}::{document.relative_path}",
            vector_store_attachment_status='indexed' if not error else 'failed',
            uploaded_at=indexed_at,
            indexed_at=indexed_at,
            last_indexed_at=indexed_at,
            last_synced_at=indexed_at,
            parser_name=parser_name,
            extracted_char_count=extracted_char_count,
            chunk_count=chunk_count,
            sync_state=sync_state,
            error=error,
        )
        manifest.documents[document.relative_path] = record


# ???????????????????????? manifest?
def sync_knowledge_base(config: SyncConfig) -> SyncSummary:
    config.source_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    manifest = KnowledgeBaseManifest.load(config.manifest_path)
    manifest.source_root = str(config.source_dir)
    manifest.index_path = str(config.index_path)
    manifest.knowledge_base_name = config.knowledge_base_name
    manifest.vector_store_name = config.knowledge_base_name
    manifest.chunk_size_chars = config.chunk_size_chars
    manifest.chunk_overlap_chars = config.chunk_overlap_chars

    documents = scan_documents(config.source_dir, config.allowed_extensions)
    total_files = len(documents)
    previous_documents = manifest.documents.copy()
    skipped_files = sum(
        1
        for document in documents
        if (existing := previous_documents.get(document.relative_path)) and existing.sha256 == document.sha256
    )
    new_or_changed_files = total_files - skipped_files
    knowledge_base_id = ensure_knowledge_base_id(manifest, config)

    if config.dry_run:
        return SyncSummary(
            source_root=config.source_dir,
            manifest_path=config.manifest_path,
            index_path=config.index_path,
            knowledge_base_id=knowledge_base_id,
            total_files=total_files,
            new_or_changed_files=new_or_changed_files,
            skipped_files=skipped_files,
            indexed_files=0,
            failed_files=0,
            chunk_count=0,
            dry_run=True,
        )

    index = build_local_index(
        documents,
        knowledge_base_name=config.knowledge_base_name,
        source_root=config.source_dir,
        index_path=config.index_path,
        chunk_size_chars=config.chunk_size_chars,
        chunk_overlap_chars=config.chunk_overlap_chars,
        knowledge_base_id=knowledge_base_id,
    )
    save_local_index(index, config.index_path)

    now = datetime.now(timezone.utc)
    parser_summaries = {item.relative_path: item for item in index.document_summaries}
    _build_manifest_records(manifest, documents, now, parser_summaries)
    manifest.source_root = str(config.source_dir)
    manifest.index_path = str(config.index_path)
    manifest.knowledge_base_id = knowledge_base_id
    manifest.vector_store_id = knowledge_base_id
    manifest.knowledge_base_name = config.knowledge_base_name
    manifest.vector_store_name = config.knowledge_base_name
    manifest.chunk_size_chars = config.chunk_size_chars
    manifest.chunk_overlap_chars = config.chunk_overlap_chars
    manifest.document_count = len(documents)
    manifest.chunk_count = index.total_chunk_count
    manifest.last_sync_at = now
    manifest.save(config.manifest_path)

    failed_files = sum(1 for item in index.document_summaries if _infer_record_error(item.parser_name, item.chunk_count))
    indexed_files = total_files - failed_files
    return SyncSummary(
        source_root=config.source_dir,
        manifest_path=config.manifest_path,
        index_path=config.index_path,
        knowledge_base_id=knowledge_base_id,
        total_files=total_files,
        new_or_changed_files=new_or_changed_files,
        skipped_files=skipped_files,
        indexed_files=indexed_files,
        failed_files=failed_files,
        chunk_count=index.total_chunk_count,
        dry_run=False,
    )
