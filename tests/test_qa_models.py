from bmagent_rag.qa_models import BrainTumorQaEnvelope, BrainTumorQaResponse, RetrievedSnippet, QaSessionState


def test_brain_tumor_response_schema_roundtrip() -> None:
    payload = {
        'answer_type': 'differential',
        'confidence': 'medium',
        'answer_summary': '更倾向于高级别胶质瘤，但仍需结合增强和病理确认。',
        'answer_detail': '这里填入更完整的中文解释。',
        'key_points': ['结节样强化提示肿瘤活性较高', '灌注和弥散可帮助进一步区分'],
        'imaging_features': [
            {
                'feature': '环形强化',
                'sequence': 'T1增强',
                'interpretation': '提示血脑屏障破坏',
                'why_it_matters': '常用于判断肿瘤活性和坏死区域。',
            }
        ],
        'differential_diagnosis': [
            {
                'entity': '胶质母细胞瘤',
                'why_considered': '环形强化和周围水肿明显。',
                'supporting_clues': ['坏死', '不规则强化'],
                'counter_clues': ['缺少足够的多灶信息'],
                'relative_likelihood': 'high',
            }
        ],
        'sequence_meaning': [
            {
                'sequence': 'DWI',
                'role': '评估弥散受限',
                'typical_findings': ['高细胞密度区可能呈弥散受限'],
                'pitfalls': ['坏死区不一定受限'],
            }
        ],
        'evidence': [
            {
                'file_id': 'chunk_1',
                'file_name': 'tumor_mri_review.pdf',
                'excerpt': 'T1增强可显示血脑屏障破坏和病灶强化模式。',
                'supports': 'supports the enhancement pattern explanation',
            }
        ],
        'limitations': ['目前仅有单篇综述证据。'],
        'follow_up_questions': ['是否有灌注或波谱结果？'],
        'safety_note': '不能替代放射科或病理诊断。',
    }

    parsed = BrainTumorQaResponse.model_validate(payload)
    assert parsed.answer_type == 'differential'
    assert parsed.evidence[0].file_name == 'tumor_mri_review.pdf'


def test_qa_envelope_accepts_local_kb_fields() -> None:
    session = QaSessionState.model_validate(
        {
            'session_id': 'session_1',
            'title': 'demo',
            'previous_response_id': None,
            'turn_count': 0,
            'created_at': '2026-03-30T00:00:00+00:00',
            'updated_at': '2026-03-30T00:00:00+00:00',
            'turns': [],
        }
    )
    answer = BrainTumorQaResponse.model_validate(
        {
            'answer_type': 'insufficient_evidence',
            'confidence': 'low',
            'answer_summary': '证据不足。',
            'answer_detail': '本地知识库没有足够证据。',
            'key_points': [],
            'imaging_features': [],
            'differential_diagnosis': [],
            'sequence_meaning': [],
            'evidence': [],
            'limitations': ['索引内容有限。'],
            'follow_up_questions': [],
            'safety_note': '仅供参考。',
        }
    )
    snippet = RetrievedSnippet.model_validate(
        {
            'source_type': 'local_bm25',
            'file_id': 'chunk_1',
            'file_name': 'review.md',
            'snippet': 'glioblastoma often shows irregular enhancement',
            'score': 2.5,
            'page_hint': 'chunk_1',
        }
    )

    envelope = BrainTumorQaEnvelope(
        session=session,
        response_id='resp_1',
        previous_response_id=None,
        knowledge_base_id='local_kb_1',
        retrieval_queries=['glioblastoma MRI review'],
        answer=answer,
        retrieved_snippets=[snippet],
    )

    assert envelope.knowledge_base_id == 'local_kb_1'
    assert envelope.retrieved_snippets[0].source_type == 'local_bm25'