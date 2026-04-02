"""?????????????????

?? Pydantic ?????? qa_api.py?qa_service.py???????
????? Schema ???????????????????????
?????????????
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field



AnswerType = Literal[
    'lesion_pattern',
    'differential',
    'sequence_meaning',
    'paper_summary',
    'mixed',
    'insufficient_evidence',
]
ConfidenceLevel = Literal['low', 'medium', 'high']
SnippetSourceType = Literal['local_bm25', 'model_evidence', 'file_search_result', 'file_citation']
SyncState = Literal['indexed', 'failed', 'pending']


class ImagingFeature(BaseModel):
    """???? MRI ??????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    feature: str = Field(description='MRI feature or observation.')
    sequence: str = Field(description='Relevant sequence or modality.')
    interpretation: str = Field(description='What the feature suggests.')
    why_it_matters: str = Field(description='Why the feature is useful in imaging interpretation.')


class DifferentialDiagnosisItem(BaseModel):
    """????????????????????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    entity: str = Field(description='Differential diagnosis entity.')
    why_considered: str = Field(description='Why this entity is considered.')
    supporting_clues: list[str] = Field(description='Imaging clues that support this entity.')
    counter_clues: list[str] = Field(description='Imaging clues that argue against this entity.')
    relative_likelihood: ConfidenceLevel = Field(description='Relative likelihood among the listed items.')


class SequenceMeaningItem(BaseModel):
    """???? MRI ???????????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    sequence: str = Field(description='MRI sequence name.')
    role: str = Field(description='Main clinical or imaging role of the sequence.')
    typical_findings: list[str] = Field(description='Typical findings highlighted by this sequence.')
    pitfalls: list[str] = Field(description='Common interpretation pitfalls.')


class EvidenceReference(BaseModel):
    """???????????????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    file_id: str | None = Field(default=None, description='Local chunk id when available.')
    file_name: str = Field(description='Document name shown in the local knowledge base.')
    excerpt: str = Field(description='Short excerpt or tight paraphrase of the supporting evidence.')
    supports: str = Field(description='What this evidence supports in the answer.')


class RetrievedSnippet(BaseModel):
    """?????????????????????????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    source_type: SnippetSourceType = Field(description='Where the snippet came from.')
    file_id: str | None = Field(default=None, description='Local chunk id when available.')
    file_name: str = Field(description='Document name shown in the local knowledge base.')
    snippet: str = Field(description='Actual retrieved text or a short excerpt.')
    score: float | None = Field(default=None, description='Retrieval score when available.')
    page_hint: str | None = Field(default=None, description='Chunk or page hint when available.')


class BrainTumorQaResponse(BaseModel):
    """?????? Schema??????????????????????? API ??????????
    """
    model_config = ConfigDict(extra='forbid')

    answer_type: AnswerType = Field(description='Question category.')
    confidence: ConfidenceLevel = Field(description='Overall answer confidence.')
    answer_summary: str = Field(description='Short answer summary in Chinese.')
    answer_detail: str = Field(description='Expanded explanation in Chinese.')
    key_points: list[str] = Field(description='Important takeaways.')
    imaging_features: list[ImagingFeature] = Field(description='Imaging features supported by the evidence.')
    differential_diagnosis: list[DifferentialDiagnosisItem] = Field(description='Differential diagnosis list.')
    sequence_meaning: list[SequenceMeaningItem] = Field(description='Meaning of MRI sequences mentioned in the question.')
    evidence: list[EvidenceReference] = Field(description='Grounding evidence from retrieved local chunks.')
    limitations: list[str] = Field(description='Evidence gaps or uncertainty statements.')
    follow_up_questions: list[str] = Field(description='Useful next questions for the user.')
    safety_note: str = Field(description='Safety note that avoids definitive diagnosis.')


class BrainTumorQaRequest(BaseModel):
    """??????????????????????????????top-k ???????? query rewrite?
    """
    model_config = ConfigDict(extra='forbid')

    session_id: str | None = Field(default=None, description='Optional session id for multi-turn conversations.')
    question: str = Field(min_length=1)
    previous_response_id: str | None = Field(default=None, description='Optional response id override for multi-turn generation.')
    knowledge_base_id: str | None = Field(default=None, description='Optional local knowledge base id to validate against the active index.')
    max_num_results: int = Field(default=5, ge=1, le=20)
    use_query_rewrite: bool = True


class QaTurnRecord(BaseModel):
    """????????????????????????????response_id ???????????
    """
    model_config = ConfigDict(extra='forbid')

    turn_index: int
    question: str
    response_id: str
    previous_response_id: str | None
    answer_summary: str
    answer_type: AnswerType
    retrieval_queries: list[str] = Field(default_factory=list)
    created_at: datetime


class QaSessionState(BaseModel):
    """??????????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    session_id: str
    title: str | None = None
    previous_response_id: str | None = None
    turn_count: int = 0
    created_at: datetime
    updated_at: datetime
    turns: list[QaTurnRecord] = Field(default_factory=list)


class BrainTumorQaEnvelope(BaseModel):
    """??????????????????????????????????????
    """
    model_config = ConfigDict(extra='forbid')

    session: QaSessionState
    response_id: str
    previous_response_id: str | None
    knowledge_base_id: str | None = None
    retrieval_queries: list[str] = Field(default_factory=list)
    answer: BrainTumorQaResponse
    retrieved_snippets: list[RetrievedSnippet] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    """?????????????????
    """
    model_config = ConfigDict(extra='forbid')

    session_id: str | None = None
    title: str | None = None


class KnowledgeBaseDocumentStatus(BaseModel):
    """?????????????????
    """
    model_config = ConfigDict(extra='forbid')

    relative_path: str
    absolute_path: str
    size_bytes: int
    sha256: str
    mime_type: str | None = None
    parser_name: str | None = None
    extracted_char_count: int = 0
    chunk_count: int = 0
    last_indexed_at: datetime | None = None
    last_synced_at: datetime | None = None
    sync_state: SyncState = 'pending'
    error: str | None = None


class KnowledgeBaseStatus(BaseModel):
    """GET /api/kb/status ?????????????
    """
    model_config = ConfigDict(extra='forbid')

    source_root: str
    manifest_path: str
    index_path: str
    knowledge_base_name: str
    knowledge_base_id: str | None = None
    source_dir_exists: bool
    state_dir_exists: bool
    total_documents: int
    indexed_documents: int
    pending_documents: int
    failed_documents: int
    chunk_count: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    last_sync_at: datetime | None = None
    updated_at: datetime | None = None
    documents: list[KnowledgeBaseDocumentStatus] = Field(default_factory=list)


class KnowledgeBaseSyncRequest(BaseModel):
    """??????? dry-run ?????????
    """
    model_config = ConfigDict(extra='forbid')

    source_dir: str | None = None
    state_dir: str | None = None
    manifest_path: str | None = None
    index_path: str | None = None
    knowledge_base_name: str | None = None
    knowledge_base_id: str | None = None
    chunk_size_chars: int | None = Field(default=None, ge=200, le=10000)
    chunk_overlap_chars: int | None = Field(default=None, ge=0, le=5000)
    dry_run: bool = False


class KnowledgeBaseSyncResponse(BaseModel):
    """????????????????
    """
    model_config = ConfigDict(extra='forbid')

    source_root: str
    manifest_path: str
    index_path: str
    knowledge_base_id: str
    total_files: int
    new_or_changed_files: int
    skipped_files: int
    indexed_files: int
    failed_files: int
    chunk_count: int
    dry_run: bool