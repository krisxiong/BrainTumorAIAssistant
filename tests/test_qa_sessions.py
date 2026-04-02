from bmagent_rag.qa_models import BrainTumorQaResponse
from bmagent_rag.qa_service import QaSessionStore


def _make_answer() -> BrainTumorQaResponse:
    return BrainTumorQaResponse(
        answer_type='differential',
        confidence='medium',
        answer_summary='更倾向于高级别胶质瘤。',
        answer_detail='这里填入更完整的中文解释。',
        key_points=['需要结合增强和病理。'],
        imaging_features=[],
        differential_diagnosis=[],
        sequence_meaning=[],
        evidence=[],
        limitations=['证据不足。'],
        follow_up_questions=[],
        safety_note='不能替代临床判断。',
    )


def test_session_store_roundtrip(tmp_path) -> None:
    store = QaSessionStore(tmp_path)
    session = store.create(title='demo')
    loaded = store.load(session.session_id)

    assert loaded.session_id == session.session_id
    assert loaded.title == 'demo'

    updated = store.record_turn(loaded, '问题是什么？', 'resp_1', _make_answer(), ['glioblastoma MRI review'])
    assert updated.turn_count == 1
    assert updated.previous_response_id == 'resp_1'
    assert updated.turns[0].response_id == 'resp_1'
    assert updated.turns[0].question == '问题是什么？'
    assert updated.turns[0].retrieval_queries == ['glioblastoma MRI review']