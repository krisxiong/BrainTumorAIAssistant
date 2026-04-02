from __future__ import annotations

import hashlib
from pathlib import Path

from bmagent_rag.config import build_config
from bmagent_rag.local_rag import load_local_index, search_local_index
from bmagent_rag.manifest import KnowledgeBaseManifest
from bmagent_rag.sync import scan_documents, sync_knowledge_base


def test_scan_documents_hashes_and_filters(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    nested = source / 'nested'
    nested.mkdir(parents=True)
    (source / 'keep.txt').write_text('alpha', encoding='utf-8')
    (nested / 'keep.md').write_text('beta', encoding='utf-8')
    (source / 'skip.bin').write_bytes(b'ignored')

    documents = scan_documents(source, ['.txt', '.md'])

    assert [doc.relative_path for doc in documents] == ['keep.txt', 'nested/keep.md']
    expected_hashes = {
        'keep.txt': hashlib.sha256(b'alpha').hexdigest(),
        'nested/keep.md': hashlib.sha256(b'beta').hexdigest(),
    }
    assert {doc.relative_path: doc.sha256 for doc in documents} == expected_hashes


def test_sync_builds_local_manifest_and_searchable_index(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    state = tmp_path / 'state'
    source.mkdir()
    state.mkdir()
    (source / 'paper1.md').write_text(
        'Glioblastoma MRI review. Ring enhancement and necrosis are common. Perfusion is often elevated.',
        encoding='utf-8',
    )
    (source / 'paper2.txt').write_text(
        'Meningioma often shows dural tail sign and extra-axial location on MRI.',
        encoding='utf-8',
    )

    config = build_config(
        source_dir=source,
        state_dir=state,
        manifest_path=state / 'manifest.json',
        index_path=state / 'local_index.json',
        knowledge_base_name='brain-tumor-mri-kb-local',
        allowed_extensions=['.md', '.txt'],
        chunk_size_chars=80,
        chunk_overlap_chars=20,
    )

    summary = sync_knowledge_base(config)
    assert summary.total_files == 2
    assert summary.indexed_files == 2
    assert summary.failed_files == 0
    assert summary.chunk_count > 0

    manifest = KnowledgeBaseManifest.load(config.manifest_path)
    assert manifest.knowledge_base_name == 'brain-tumor-mri-kb-local'
    assert manifest.chunk_count == summary.chunk_count
    assert manifest.documents['paper1.md'].sync_state == 'indexed'

    index = load_local_index(config.index_path)
    hits = search_local_index(index, 'glioblastoma ring enhancement perfusion', top_k=3)
    assert hits
    assert hits[0].file_name == 'paper1.md'


def test_sync_dry_run_reports_pending_without_writing_index(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    state = tmp_path / 'state'
    source.mkdir()
    state.mkdir()
    (source / 'paper1.md').write_text('Brain tumor MRI review', encoding='utf-8')

    config = build_config(
        source_dir=source,
        state_dir=state,
        manifest_path=state / 'manifest.json',
        index_path=state / 'local_index.json',
        allowed_extensions=['.md'],
        dry_run=True,
    )

    summary = sync_knowledge_base(config)
    assert summary.dry_run is True
    assert summary.new_or_changed_files == 1
    assert summary.indexed_files == 0
    assert not config.index_path.exists()