from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from bmagent_rag.config import build_config
from bmagent_rag.local_rag import LocalChunk, LocalKnowledgeBaseIndex, tokenize, search_local_index
from bmagent_rag.manifest import KnowledgeBaseManifest
from bmagent_rag.qa_config import QaConfig
from bmagent_rag.qa_models import BrainTumorQaRequest
from bmagent_rag.qa_service import BrainTumorQaService, QaSessionStore
from bmagent_rag.sync import sync_knowledge_base


class _FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            text = '这不是合法 JSON'
        elif self.calls == 2:
            text = '依然不是 JSON'
        else:
            text = (
                '胶质母细胞瘤在 MRI 上常表现为不规则强化、中央坏死和明显周围水肿。\n\n'
                '结合检索证据，这类病灶通常呈侵袭性生长，增强后可见环形或不均匀强化，周围白质常伴较明显占位效应。\n\n'
                '现有证据来自本地命中的综述和原始研究片段，仍需结合具体序列和病理结果解读。'
            )
        return SimpleNamespace(
            id=f'resp_{self.calls}',
            output_text=text,
            output=[SimpleNamespace(type='message', content=[SimpleNamespace(type='output_text', text=text)])],
        )


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _ConservativeThenSupplementResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            text = '\u8fd9\u4e0d\u662f\u5408\u6cd5 JSON'
        elif self.calls == 2:
            text = '\u4f9d\u7136\u4e0d\u662f JSON'
        elif self.calls == 3:
            text = '\u7b80\u77ed\u7ed3\u8bba\uff1a\u8bc1\u636e\u672a\u76f4\u63a5\u63cf\u8ff0\u80f6\u8d28\u6bcd\u7ec6\u80de\u7624\u5728 MRI \u4e0a\u7684\u5178\u578b\u8868\u73b0\uff0c\u6682\u65e0\u6cd5\u660e\u786e\u56de\u7b54\u3002'
        else:
            text = (
                '\u8bc1\u636e\u76f4\u63a5\u652f\u6301\uff1a\u5f53\u524d\u672c\u5730\u547d\u4e2d\u6587\u732e\u4e3b\u8981\u8ba8\u8bba\u80f6\u8d28\u6bcd\u7ec6\u80de\u7624\u76f8\u5173 MRI \u5206\u6790\u4e0e\u5206\u5b50\u75c5\u7406\u8bc4\u4f30\uff0c\u76f4\u63a5\u63cf\u8ff0\u5178\u578b\u5f71\u50cf\u8868\u73b0\u7684\u8bc1\u636e\u6709\u9650\u3002\n\n'
                '\u901a\u7528\u77e5\u8bc6\u8865\u5145\uff08\u975e\u672c\u5730\u8bc1\u636e\u76f4\u63a5\u652f\u6301\uff09\uff1a\u80f6\u8d28\u6bcd\u7ec6\u80de\u7624\u5728 MRI \u4e0a\u5e38\u89c1\u8868\u73b0\u5305\u62ec\u4e0d\u89c4\u5219\u73af\u5f62\u6216\u4e0d\u5747\u5300\u5f3a\u5316\u3001\u4e2d\u5fc3\u574f\u6b7b\u3001\u660e\u663e\u5468\u56f4 vasogenic edema\u3001\u5360\u4f4d\u6548\u5e94\uff0c\u4ee5\u53ca\u6cbf\u767d\u8d28\u675f\u6d78\u6da6\u5bfc\u81f4\u8fb9\u754c\u4e0d\u6e05\u3002\u90e8\u5206\u75c5\u4f8b\u53ef\u89c1\u80fc\u80dd\u4f53\u53d7\u7d2f\u6216\u8de8\u4e2d\u7ebf\u751f\u957f\u3002'
            )
        return SimpleNamespace(
            id=f'resp_{self.calls}',
            output_text=text,
            output=[SimpleNamespace(type='message', content=[SimpleNamespace(type='output_text', text=text)])],
        )


class _ConservativeThenSupplementClient:
    def __init__(self) -> None:
        self.responses = _ConservativeThenSupplementResponses()


class _NoHitGeneralResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        text = (
            '\u8bc1\u636e\u76f4\u63a5\u652f\u6301\uff1a\u5f53\u524d\u6ca1\u6709\u76f4\u63a5\u8bc1\u636e\u3002\n\n'
            '\u901a\u7528\u77e5\u8bc6\u8865\u5145\uff08\u975e\u672c\u5730\u8bc1\u636e\u76f4\u63a5\u652f\u6301\uff09\uff1a\u80f6\u8d28\u6bcd\u7ec6\u80de\u7624\u7684\u5178\u578b MRI \u8868\u73b0\u901a\u5e38\u5305\u62ec\u4e0d\u89c4\u5219\u5f3a\u5316\u3001\u4e2d\u5fc3\u574f\u6b7b\u3001\u660e\u663e\u5468\u56f4\u6c34\u80bf\u548c\u8f83\u5f3a\u5360\u4f4d\u6548\u5e94\u3002'
        )
        return SimpleNamespace(
            id=f'resp_{self.calls}',
            output_text=text,
            output=[SimpleNamespace(type='message', content=[SimpleNamespace(type='output_text', text=text)])],
        )


class _NoHitGeneralClient:
    def __init__(self) -> None:
        self.responses = _NoHitGeneralResponses()


def _make_chunk(text: str, chunk_id: str) -> LocalChunk:
    tokens = tokenize(text)
    return LocalChunk(
        chunk_id=chunk_id,
        document_relative_path='paper.md',
        document_sha256='sha-demo',
        chunk_index=0,
        text=text,
        token_count=len(tokens),
        character_count=len(text),
        term_frequencies=dict(Counter(tokens)),
    )


def test_search_local_index_filters_pdf_metadata_noise() -> None:
    good_text = 'Glioblastoma MRI review. Irregular ring enhancement, necrosis, and vasogenic edema are common findings.'
    noise_text = '%PDF-1.4 xref endobj rdf:Bag dc:format pdf:Producer CrossMarkDomains 12345'
    good_chunk = _make_chunk(good_text, 'good::0000')
    noise_chunk = _make_chunk(noise_text, 'noise::0000')
    index = LocalKnowledgeBaseIndex(
        knowledge_base_id='local-bm25://demo',
        knowledge_base_name='demo',
        source_root='source',
        index_path='index.json',
        chunk_size_chars=1400,
        chunk_overlap_chars=250,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        document_summaries=[],
        chunks=[noise_chunk, good_chunk],
        document_frequency=dict(Counter(set(tokenize(noise_text)) | set(tokenize(good_text)))),
        average_chunk_length=(noise_chunk.token_count + good_chunk.token_count) / 2,
        total_chunk_count=2,
        total_token_count=noise_chunk.token_count + good_chunk.token_count,
    )

    hits = search_local_index(index, 'glioblastoma MRI enhancement edema', top_k=3)

    assert hits
    assert hits[0].chunk_id == 'good::0000'
    assert all('noise::0000' != hit.chunk_id for hit in hits)


def test_generate_answer_falls_back_to_plain_text_wrapped_response() -> None:
    service = BrainTumorQaService(
        QaConfig(openai_api_key='test-key', openai_model='demo-model'),
        client=_FakeClient(),
    )
    chunk = _make_chunk(
        'Glioblastoma often shows irregular enhancement, central necrosis, and surrounding vasogenic edema on MRI.',
        'paper.md::0000',
    )
    index = LocalKnowledgeBaseIndex(
        knowledge_base_id='local-bm25://demo',
        knowledge_base_name='demo',
        source_root='source',
        index_path='index.json',
        chunk_size_chars=1400,
        chunk_overlap_chars=250,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        document_summaries=[],
        chunks=[chunk],
        document_frequency=dict(Counter(set(tokenize(chunk.text)))),
        average_chunk_length=chunk.token_count,
        total_chunk_count=1,
        total_token_count=chunk.token_count,
    )
    search_hit = search_local_index(index, 'glioblastoma MRI enhancement necrosis edema', top_k=1)[0]

    answer, response_id = service._generate_answer(
        question='胶质母细胞瘤在影像上的表现是什么？',
        hits=[search_hit],
        previous_response_id=None,
    )

    assert response_id == 'resp_3'
    assert answer.answer_summary
    assert '不规则强化' in answer.answer_detail
    assert answer.limitations
    assert '自由文本生成' in answer.limitations[0]
    assert answer.evidence[0].file_id == 'paper.md::0000'


def test_sync_marks_pdf_missing_dependency_as_failed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / 'source'
    state = tmp_path / 'state'
    source.mkdir()
    state.mkdir()
    (source / 'paper.pdf').write_bytes(b'%PDF-demo')

    def fake_read_text_from_path(path: Path):
        return '', 'pdf_missing_dependency'

    monkeypatch.setattr('bmagent_rag.local_rag.read_text_from_path', fake_read_text_from_path)

    config = build_config(
        source_dir=source,
        state_dir=state,
        manifest_path=state / 'manifest.json',
        index_path=state / 'local_index.json',
        allowed_extensions=['.pdf'],
    )

    summary = sync_knowledge_base(config)
    manifest = KnowledgeBaseManifest.load(config.manifest_path)

    assert summary.failed_files == 1
    assert summary.indexed_files == 0
    assert manifest.documents['paper.pdf'].sync_state == 'failed'
    assert 'pypdf' in (manifest.documents['paper.pdf'].error or '')

def test_generate_answer_requests_general_knowledge_supplement_when_text_is_too_conservative() -> None:
    service = BrainTumorQaService(
        QaConfig(openai_api_key='test-key', openai_model='demo-model'),
        client=_ConservativeThenSupplementClient(),
    )
    chunk = _make_chunk(
        'Glioblastoma MRI research may discuss enhancement and edema, but this chunk is not a full textbook description.',
        'paper.md::0000',
    )
    index = LocalKnowledgeBaseIndex(
        knowledge_base_id='local-bm25://demo',
        knowledge_base_name='demo',
        source_root='source',
        index_path='index.json',
        chunk_size_chars=1400,
        chunk_overlap_chars=250,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        document_summaries=[],
        chunks=[chunk],
        document_frequency=dict(Counter(set(tokenize(chunk.text)))),
        average_chunk_length=chunk.token_count,
        total_chunk_count=1,
        total_token_count=chunk.token_count,
    )
    search_hit = search_local_index(index, 'glioblastoma MRI features enhancement edema', top_k=1)[0]

    answer, response_id = service._generate_answer(
        question='\u80f6\u8d28\u6bcd\u7ec6\u80de\u7624\u5728\u5f71\u50cf\u4e0a\u7684\u8868\u73b0\u662f\u4ec0\u4e48\uff1f',
        hits=[search_hit],
        previous_response_id=None,
    )

    assert response_id == 'resp_4'
    assert '\u901a\u7528\u77e5\u8bc6\u8865\u5145\uff08\u975e\u672c\u5730\u8bc1\u636e\u76f4\u63a5\u652f\u6301\uff09' in answer.answer_detail
    assert any('\u901a\u7528\u77e5\u8bc6\u8865\u5145' in item for item in answer.limitations)


def test_answer_uses_general_knowledge_when_no_hits(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    index_path = state_dir / 'local_index.json'
    manifest_path = state_dir / 'manifest.json'
    empty_index = LocalKnowledgeBaseIndex(
        knowledge_base_id='local-bm25://demo',
        knowledge_base_name='demo',
        source_root=str(tmp_path / 'source'),
        index_path=str(index_path),
        chunk_size_chars=1400,
        chunk_overlap_chars=250,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        document_summaries=[],
        chunks=[],
        document_frequency={},
        average_chunk_length=0.0,
        total_chunk_count=0,
        total_token_count=0,
    )
    import json
    index_path.write_text(json.dumps(empty_index.to_payload(), ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path.write_text(
        KnowledgeBaseManifest(knowledge_base_id='local-bm25://demo', knowledge_base_name='demo').model_dump_json(indent=2),
        encoding='utf-8',
    )

    config = build_config(
        source_dir=tmp_path / 'source',
        state_dir=state_dir,
        manifest_path=manifest_path,
        index_path=index_path,
        knowledge_base_name='demo',
    )
    monkeypatch.setattr('bmagent_rag.qa_service.build_config', lambda: config)

    (tmp_path / 'source').mkdir()
    service = BrainTumorQaService(
        QaConfig(openai_api_key='test-key', openai_model='demo-model'),
        client=_NoHitGeneralClient(),
        session_store=QaSessionStore(tmp_path / 'sessions'),
    )
    envelope = service.answer(
        BrainTumorQaRequest(
            question='\u80f6\u8d28\u6bcd\u7ec6\u80de\u7624\u5728 MRI \u4e0a\u7684\u5178\u578b\u8868\u73b0\u662f\u4ec0\u4e48\uff1f',
            use_query_rewrite=False,
            knowledge_base_id='local-bm25://demo',
        )
    )

    assert envelope.answer.answer_summary
    assert '\u901a\u7528\u77e5\u8bc6\u8865\u5145\uff08\u975e\u672c\u5730\u8bc1\u636e\u76f4\u63a5\u652f\u6301\uff09' in envelope.answer.answer_detail
    assert envelope.answer.evidence == []
    assert any('\u6ca1\u6709\u547d\u4e2d\u76f4\u63a5\u8bc1\u636e' in item for item in envelope.answer.limitations)
