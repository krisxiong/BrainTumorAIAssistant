from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

import bmagent_rag.qa_api as qa_api
from bmagent_rag.qa_models import (
    BrainTumorQaEnvelope,
    BrainTumorQaResponse,
    QaSessionState,
)


client = TestClient(qa_api.app)


def _make_session(session_id: str = 'session-demo') -> QaSessionState:
    now = datetime.now(timezone.utc)
    return QaSessionState(
        session_id=session_id,
        title='demo',
        previous_response_id=None,
        turn_count=0,
        created_at=now,
        updated_at=now,
        turns=[],
    )


def _make_answer() -> BrainTumorQaResponse:
    return BrainTumorQaResponse(
        answer_type='sequence_meaning',
        confidence='low',
        answer_summary='该序列主要用于判断出血和钙化相关信号。',
        answer_detail='这里填入更完整的中文解释。',
        key_points=[],
        imaging_features=[],
        differential_diagnosis=[],
        sequence_meaning=[],
        evidence=[],
        limitations=['证据不足。'],
        follow_up_questions=[],
        safety_note='仅供教育和检索辅助使用。',
    )


def test_health_and_kb_status_routes_exist() -> None:
    health = client.get('/healthz')
    kb_status = client.get('/api/kb/status')

    assert health.status_code == 200
    assert kb_status.status_code == 200
    assert '/api/qa' in {route.path for route in qa_api.app.routes}
    assert '/api/sessions' in {route.path for route in qa_api.app.routes}


def test_session_create_and_get_roundtrip() -> None:
    created = client.post('/api/sessions', json={'title': 'demo session'})
    assert created.status_code == 200
    session_id = created.json()['session_id']

    fetched = client.get(f'/api/sessions/{session_id}')
    assert fetched.status_code == 200
    assert fetched.json()['session_id'] == session_id


def test_qa_endpoint_uses_service(monkeypatch) -> None:
    session = _make_session()
    answer = _make_answer()
    envelope = BrainTumorQaEnvelope(
        session=session,
        response_id='resp_123',
        previous_response_id=None,
        knowledge_base_id='local_kb_123',
        retrieval_queries=['SWI meaning'],
        answer=answer,
        retrieved_snippets=[],
    )

    def fake_answer(request):
        return envelope

    monkeypatch.setattr(qa_api.qa_service, 'answer', fake_answer)

    response = client.post('/api/qa', json={'question': 'SWI 有什么意义？', 'knowledge_base_id': 'local_kb_123'})
    assert response.status_code == 200
    body = response.json()
    assert body['response_id'] == 'resp_123'
    assert body['knowledge_base_id'] == 'local_kb_123'
    assert body['answer']['answer_type'] == 'sequence_meaning'