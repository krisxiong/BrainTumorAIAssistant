# 本文件是本地 RAG 的核心编排层，qa_api.py 会调用这里完成检索、生成、会话状态和证据组装。
# 这个文件也是你读懂整个问答 workflow 的关键入口。
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from openai import OpenAI

from .config import build_config
from .local_rag import SearchHit, load_local_index, search_local_index
from .manifest import KnowledgeBaseManifest
from .qa_config import QaConfig
from .qa_models import (
    BrainTumorQaEnvelope,
    BrainTumorQaRequest,
    BrainTumorQaResponse,
    EvidenceReference,
    QaSessionState,
    QaTurnRecord,
    RetrievedSnippet,
)
from .qa_prompts import build_answer_system_prompt, build_query_rewrite_prompt


# 会话存储类，负责把多轮问答上下文落盘为 JSON 文件。
class QaSessionStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def create(self, session_id: str | None = None, title: str | None = None) -> QaSessionState:
        resolved_session_id = self._resolve_session_id(session_id or uuid4().hex)
        now = datetime.now(timezone.utc)
        state = QaSessionState(
            session_id=resolved_session_id,
            title=title,
            previous_response_id=None,
            turn_count=0,
            created_at=now,
            updated_at=now,
            turns=[],
        )
        self.save(state)
        return state

    def get_or_create(self, session_id: str | None = None, title: str | None = None) -> QaSessionState:
        if session_id:
            try:
                state = self.load(session_id)
                if title and not state.title:
                    state.title = title
                    self.save(state)
                return state
            except FileNotFoundError:
                return self.create(session_id=session_id, title=title)
        return self.create(title=title)

    def load(self, session_id: str) -> QaSessionState:
        path = self._session_path(self._resolve_session_id(session_id))
        if not path.exists():
            raise FileNotFoundError(session_id)
        return QaSessionState.model_validate_json(path.read_text(encoding='utf-8'))

    def save(self, state: QaSessionState) -> None:
        path = self._session_path(state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix('.json.tmp')
        tmp_path.write_text(state.model_dump_json(indent=2), encoding='utf-8')
        tmp_path.replace(path)

    def record_turn(
        self,
        state: QaSessionState,
        question: str,
        response_id: str,
        answer: BrainTumorQaResponse,
        retrieval_queries: list[str],
    ) -> QaSessionState:
        now = datetime.now(timezone.utc)
        turn = QaTurnRecord(
            turn_index=state.turn_count + 1,
            question=question,
            response_id=response_id,
            previous_response_id=state.previous_response_id,
            answer_summary=answer.answer_summary,
            answer_type=answer.answer_type,
            retrieval_queries=retrieval_queries,
            created_at=now,
        )
        state.turns.append(turn)
        state.turn_count += 1
        state.previous_response_id = response_id
        state.updated_at = now
        if not state.title:
            state.title = question[:32].strip()
        self.save(state)
        return state

    def _resolve_session_id(self, session_id: str) -> str:
        normalized = Path(session_id).name
        if normalized != session_id:
            raise HTTPException(status_code=400, detail='session_id contains unsupported path characters.')
        return normalized

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f'{session_id}.json'


# 问答服务类，把检索、模型生成、会话管理和结构化输出串成一条链路。
class BrainTumorQaService:
    def __init__(self, config: QaConfig, client: OpenAI | None = None, session_store: QaSessionStore | None = None) -> None:
        self.config = config
        self.client = client or self._build_client(config)
        self.session_store = session_store or QaSessionStore(Path('storage') / 'sessions')

    def create_session(self, session_id: str | None = None, title: str | None = None) -> QaSessionState:
        return self.session_store.get_or_create(session_id=session_id, title=title)

    def load_session(self, session_id: str) -> QaSessionState:
        return self.session_store.load(session_id)

    # 问答主流程：读取知识库、检索 top-k、调用模型生成并落盘会话记录。
    def answer(self, request: BrainTumorQaRequest) -> BrainTumorQaEnvelope:
        sync_config = build_config()
        manifest = KnowledgeBaseManifest.load(sync_config.manifest_path)
        if not sync_config.index_path.exists():
            raise HTTPException(status_code=400, detail='本地知识库索引不存在，请先执行 /api/kb/sync。')
        if request.knowledge_base_id and manifest.knowledge_base_id and request.knowledge_base_id != manifest.knowledge_base_id:
            raise HTTPException(status_code=404, detail='指定的 knowledge_base_id 与当前本地知识库不匹配。')

        knowledge_base_id = manifest.knowledge_base_id or sync_config.knowledge_base_id or 'local_kb_unknown'
        index = load_local_index(sync_config.index_path)
        session = self.session_store.get_or_create(session_id=request.session_id)
        effective_previous_response_id = request.previous_response_id or session.previous_response_id

        retrieval_queries = self._plan_retrieval_queries(request.question, request.use_query_rewrite)
        combined_query = ' '.join(dict.fromkeys([request.question, *retrieval_queries]))
        hits = search_local_index(index, combined_query, top_k=request.max_num_results)
        retrieved_snippets = self._hits_to_snippets(hits)

        if not hits:
            response_id = f'local_{uuid4().hex[:12]}'
            plain_text, response_id = self._generate_plain_text_answer(
                question=request.question,
                hits=[],
                previous_response_id=effective_previous_response_id,
                response_id=response_id,
            )
            if plain_text:
                answer = self._build_text_fallback_answer(request.question, [], plain_text)
            else:
                answer = self._build_insufficient_answer(request.question)
            session = self.session_store.record_turn(session, request.question, response_id, answer, retrieval_queries)
            return BrainTumorQaEnvelope(
                session=session,
                response_id=response_id,
                previous_response_id=effective_previous_response_id,
                knowledge_base_id=knowledge_base_id,
                retrieval_queries=retrieval_queries,
                answer=answer,
                retrieved_snippets=[],
            )

        answer, response_id = self._generate_answer(
            question=request.question,
            hits=hits,
            previous_response_id=effective_previous_response_id,
        )
        if not answer.evidence:
            answer.evidence = self._hits_to_evidence(hits)
        session = self.session_store.record_turn(session, request.question, response_id, answer, retrieval_queries)

        return BrainTumorQaEnvelope(
            session=session,
            response_id=response_id,
            previous_response_id=effective_previous_response_id,
            knowledge_base_id=knowledge_base_id,
            retrieval_queries=retrieval_queries,
            answer=answer,
            retrieved_snippets=retrieved_snippets,
        )

    def _build_client(self, config: QaConfig) -> OpenAI | None:
        if not config.openai_api_key:
            return None
        return OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url or None)

    # 把用户问题改写成更适合检索的查询词列表。
    def _plan_retrieval_queries(self, question: str, use_query_rewrite: bool) -> list[str]:
        if not use_query_rewrite or self.client is None:
            return [question]

        try:
            response = self.client.responses.create(
                model=self.config.openai_model,
                input=build_query_rewrite_prompt(question),
                max_output_tokens=180,
            )
            text = self._extract_output_text(response)
            payload = self._extract_json_object(text)
            if isinstance(payload, dict):
                queries = [str(item).strip() for item in payload.get('queries', []) if str(item).strip()]
                if queries:
                    return list(dict.fromkeys([question, *queries]))[:4]
        except Exception:
            pass
        return [question]

    # 将 top-k 证据组装为 prompt，优先尝试严格结构化输出；若兼容网关不稳定，则退到自由文本生成再包装成结构化结果。
    def _generate_answer(
        self,
        *,
        question: str,
        hits: list[SearchHit],
        previous_response_id: str | None,
    ) -> tuple[BrainTumorQaResponse, str]:
        response_id = f'local_{uuid4().hex[:12]}'
        if self.client is None:
            return self._build_retrieval_only_answer(question, hits), response_id

        context_block = self._build_context_block(hits)
        prompt = (
            f'用户问题:\n{question}\n\n'
            f'本地检索证据:\n{context_block}\n\n'
            '请基于以上证据作答，并严格输出 JSON。'
        )

        try:
            response = self.client.responses.create(
                model=self.config.openai_model,
                instructions=build_answer_system_prompt(),
                input=prompt,
                previous_response_id=previous_response_id,
                reasoning={'effort': self.config.openai_reasoning_effort},
                text={
                    'verbosity': self.config.openai_text_verbosity,
                    'format': {
                        'type': 'json_schema',
                        'name': 'brain_tumor_mri_qa_response',
                        'strict': True,
                        'schema': BrainTumorQaResponse.model_json_schema(),
                    },
                },
                max_output_tokens=self.config.openai_max_output_tokens,
            )
            response_id = getattr(response, 'id', response_id)
            raw_text = self._extract_output_text(response)
            if raw_text:
                try:
                    return BrainTumorQaResponse.model_validate_json(raw_text), response_id
                except Exception:
                    payload = self._extract_json_object(raw_text)
                    if isinstance(payload, dict):
                        return BrainTumorQaResponse.model_validate(payload), response_id
        except Exception:
            pass

        try:
            response = self.client.responses.create(
                model=self.config.openai_model,
                input=(
                    f'{build_answer_system_prompt()}\n\n'
                    '你必须只输出一个 JSON 对象，不能输出 markdown 或解释性文字。\n\n'
                    f'用户问题:\n{question}\n\n'
                    f'本地检索证据:\n{context_block}'
                ),
                previous_response_id=previous_response_id,
                max_output_tokens=self.config.openai_max_output_tokens,
            )
            response_id = getattr(response, 'id', response_id)
            raw_text = self._extract_output_text(response)
            payload = self._extract_json_object(raw_text)
            if isinstance(payload, dict):
                return BrainTumorQaResponse.model_validate(payload), response_id
        except Exception:
            pass

        plain_text, response_id = self._generate_plain_text_answer(
            question=question,
            hits=hits,
            previous_response_id=previous_response_id,
            response_id=response_id,
        )
        if plain_text:
            return self._build_text_fallback_answer(question, hits, plain_text), response_id

        return self._build_retrieval_only_answer(question, hits), response_id

    # 当严格 JSON 不稳定时，退化为“可读文本回答”。优先依据证据作答，证据不足时允许显式标注的通用知识补充。
    def _generate_plain_text_answer(
        self,
        *,
        question: str,
        hits: list[SearchHit],
        previous_response_id: str | None,
        response_id: str,
    ) -> tuple[str | None, str]:
        if self.client is None:
            return None, response_id

        context_block = self._build_context_block(hits) if hits else '当前没有命中的本地检索证据。'
        try:
            response = self.client.responses.create(
                model=self.config.openai_model,
                instructions=(
                    '你是脑肿瘤 MRI 问答助手。'
                    '请优先依据提供的本地检索证据回答。'
                    '如果证据只能部分回答问题，请先写“证据直接支持：”，再写“通用知识补充（非本地证据直接支持）：”。'
                    '如果当前没有检索证据，也可以基于通用医学影像知识给出简洁回答，但必须明确标注“通用知识补充（非本地证据直接支持）”。'
                    '不能把通用知识伪装成本地证据，不能给出确定性诊断或治疗建议。'
                    '请用简体中文直接输出可读答案，不要输出 JSON，不要输出 markdown 代码块。'
                ),
                input=(
                    f'用户问题:\n{question}\n\n'
                    f'本地检索证据:\n{context_block}\n\n'
                    '请直接给出中文答案。'
                ),
                previous_response_id=previous_response_id,
                max_output_tokens=self.config.openai_max_output_tokens,
            )
            response_id = getattr(response, 'id', response_id)
            text = self._cleanup_model_text(self._extract_output_text(response))
            if text and self._text_needs_general_knowledge_boost(text):
                supplemented_text, response_id = self._generate_general_knowledge_supplement(
                    question=question,
                    hits=hits,
                    previous_response_id=previous_response_id,
                    response_id=response_id,
                )
                if supplemented_text:
                    return supplemented_text, response_id
            if text:
                return text, response_id
        except Exception:
            pass
        return None, response_id

    # 当模型只给出“证据不足”式回答时，再追发一轮，要求输出带明确标签的通用知识补充。
    def _generate_general_knowledge_supplement(
        self,
        *,
        question: str,
        hits: list[SearchHit],
        previous_response_id: str | None,
        response_id: str,
    ) -> tuple[str | None, str]:
        if self.client is None:
            return None, response_id

        context_block = self._build_context_block(hits) if hits else '当前没有命中的本地检索证据。'
        try:
            response = self.client.responses.create(
                model=self.config.openai_model,
                instructions=(
                    '你是脑肿瘤 MRI 助手。'
                    '请输出两个部分。'
                    '第一部分标题固定为“证据直接支持：”，只总结当前本地证据真正支持的内容。'
                    '第二部分标题固定为“通用知识补充（非本地证据直接支持）：”，补充该问题在常见脑肿瘤 MRI 知识中的标准回答。'
                    '如果没有本地证据，第一部分写“当前没有直接证据”。'
                    '补充内容必须明确标记为非本地证据直接支持，不能伪装成检索证据。'
                    '不要输出 JSON，不要输出代码块，只输出简体中文正文。'
                ),
                input=(
                    f'用户问题:\n{question}\n\n'
                    f'本地检索证据:\n{context_block}\n\n'
                    '请给出面向医生助理场景的简洁专业回答。'
                ),
                previous_response_id=previous_response_id,
                max_output_tokens=self.config.openai_max_output_tokens,
            )
            response_id = getattr(response, 'id', response_id)
            text = self._cleanup_model_text(self._extract_output_text(response))
            if text:
                return text, response_id
        except Exception:
            pass
        return None, response_id

    # 判断模型文本是否停留在“证据不足/无法回答”，如果是，就触发一轮通用知识补充。
    def _text_needs_general_knowledge_boost(self, text: str) -> bool:
        cleaned = self._cleanup_model_text(text)
        if not cleaned:
            return False
        if '通用知识补充（非本地证据直接支持）' in cleaned:
            return False
        refusal_markers = (
            '未直接描述',
            '未直接涉及',
            '无法明确回答',
            '暂无法明确回答',
            '无法完整回答',
            '证据不足',
            '未在证据中详细说明',
            '目前证据仅侧重',
        )
        return any(marker in cleaned for marker in refusal_markers)

    # 将自由文本降级结果重新包装为统一的 BrainTumorQaResponse，前端就不需要再分支处理不同格式。
    def _build_text_fallback_answer(self, question: str, hits: list[SearchHit], generated_text: str) -> BrainTumorQaResponse:
        cleaned = self._cleanup_model_text(generated_text)
        top_files = list(dict.fromkeys(hit.file_name for hit in hits[:3]))
        key_points = self._extract_key_points(cleaned)
        if not key_points:
            key_points = [f'主要证据来源：{name}' for name in top_files] or ['当前结果由本地检索证据和模型通用知识补充共同组成。']

        limitations = ['Yunwu/OpenAI 兼容接口本轮没有稳定返回可解析的结构化 JSON，因此已降级为自由文本生成，再由应用层包装成统一结构。']
        if not hits:
            limitations.append('当前本地知识库没有命中直接证据，回答主要来自模型通用医学影像知识补充。')
        elif '通用知识补充（非本地证据直接支持）' in cleaned:
            limitations.append('回答中包含模型通用知识补充，其中该部分未被当前本地知识库直接支持。')

        return BrainTumorQaResponse(
            answer_type=self._infer_answer_type(question),
            confidence='medium' if len(hits) >= 3 else 'low',
            answer_summary=self._build_summary(cleaned),
            answer_detail=cleaned,
            key_points=key_points,
            imaging_features=[],
            differential_diagnosis=[],
            sequence_meaning=[],
            evidence=self._hits_to_evidence(hits),
            limitations=limitations,
            follow_up_questions=self._suggest_follow_up_questions(question),
            safety_note='本助手仅提供基于文献证据的影像学信息整理和必要的通用知识补充，不能替代放射科医师、神经肿瘤团队和病理结果。',
        )

    # 模型结构化生成失败时的最终兜底回答。即使模型完全不可用，也至少把命中的文档和证据暴露出来。
    def _build_retrieval_only_answer(self, question: str, hits: list[SearchHit]) -> BrainTumorQaResponse:
        top_files = list(dict.fromkeys(hit.file_name for hit in hits[:3]))
        summary = '已根据本地检索证据整理出相关信息，但模型结构化生成失败，建议优先查看证据片段。'
        detail_lines = [
            f'问题：{question}',
            f'本次命中 {len(hits)} 个本地片段。',
        ]
        if top_files:
            detail_lines.append('主要来源文档：' + '、'.join(top_files))
        detail_lines.append('当前回答为检索兜底结果，内容更偏证据摘要而不是完整结构化解读。')
        return BrainTumorQaResponse(
            answer_type='mixed',
            confidence='low',
            answer_summary=summary,
            answer_detail='\n'.join(detail_lines),
            key_points=[f'优先查看证据来源：{name}' for name in top_files] or ['本次回答基于本地 BM25 检索结果整理。'],
            imaging_features=[],
            differential_diagnosis=[],
            sequence_meaning=[],
            evidence=self._hits_to_evidence(hits),
            limitations=['Yunwu/OpenAI 兼容接口这次没有稳定返回可解析的结构化 JSON，因此使用了检索兜底回答。'],
            follow_up_questions=self._suggest_follow_up_questions(question),
            safety_note='本助手仅提供基于文献证据的影像学信息整理，不能替代放射科医师、神经肿瘤团队和病理结果。',
        )

    # 检索没有命中足够证据时的标准回答。
    def _build_insufficient_answer(self, question: str) -> BrainTumorQaResponse:
        return BrainTumorQaResponse(
            answer_type='insufficient_evidence',
            confidence='low',
            answer_summary='当前本地知识库中没有检索到足够相关的证据片段。',
            answer_detail=(
                f'问题：{question}\n'
                '这次检索没有命中足够相关的本地文档内容，因此暂时无法给出可靠的证据增强回答。'
                '建议补充教材章节、综述或更精确的肿瘤/序列关键词后再试。'
            ),
            key_points=['可以先扩充本地知识库，或换用更具体的检索问题。'],
            imaging_features=[],
            differential_diagnosis=[],
            sequence_meaning=[],
            evidence=[],
            limitations=['本地知识库覆盖范围不足，或当前问题中的关键词没有命中现有片段。'],
            follow_up_questions=self._suggest_follow_up_questions(question),
            safety_note='本助手仅提供基于本地文献证据的说明，不能替代临床诊断。',
        )

    def _hits_to_snippets(self, hits: list[SearchHit]) -> list[RetrievedSnippet]:
        return [
            RetrievedSnippet(
                source_type='local_bm25',
                file_id=hit.chunk_id,
                file_name=hit.file_name,
                snippet=hit.snippet,
                score=hit.score,
                page_hint=f'chunk {hit.chunk_index}',
            )
            for hit in hits
        ]

    def _hits_to_evidence(self, hits: list[SearchHit]) -> list[EvidenceReference]:
        return [
            EvidenceReference(
                file_id=hit.chunk_id,
                file_name=hit.file_name,
                excerpt=hit.snippet[:320],
                supports='支持回答中的相关论点。',
            )
            for hit in hits[:5]
        ]

    def _build_context_block(self, hits: list[SearchHit]) -> str:
        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            blocks.append(f'[{index}] file={hit.file_name} chunk={hit.chunk_id} score={hit.score}')
            blocks.append(hit.snippet)
            blocks.append('')
        return '\n'.join(blocks).strip()

    def _cleanup_model_text(self, text: str) -> str:
        cleaned = (text or '').strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json|text)?\s*', '', cleaned, count=1)
            cleaned = re.sub(r'\s*```$', '', cleaned, count=1)
        return cleaned.strip()

    def _build_summary(self, text: str, max_chars: int = 110) -> str:
        compact = re.sub(r'\s+', ' ', text).strip()
        if len(compact) <= max_chars:
            return compact
        clipped = compact[:max_chars].rstrip(' ，；;,:')
        if not clipped.endswith(('。', '！', '？')):
            clipped += '。'
        return clipped

    def _extract_key_points(self, text: str, max_items: int = 4) -> list[str]:
        candidates: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip().lstrip('-•0123456789. ').strip()
            if len(stripped) < 8:
                continue
            if stripped not in candidates:
                candidates.append(stripped)
            if len(candidates) >= max_items:
                break
        if candidates:
            return candidates[:max_items]

        sentences = [item.strip() for item in re.split(r'[。！？!?]\s*', text) if len(item.strip()) >= 8]
        deduped: list[str] = []
        for sentence in sentences:
            if sentence not in deduped:
                deduped.append(sentence + '。')
            if len(deduped) >= max_items:
                break
        return deduped

    def _infer_answer_type(self, question: str) -> str:
        lowered = question.lower()
        if any(keyword in question for keyword in ('鉴别', '区分', '区别', '对比')):
            return 'differential'
        if any(keyword in lowered for keyword in ('dwi', 'adc', 'flair', 'swi', 't1', 't2')) or '序列' in question:
            return 'sequence_meaning'
        if any(keyword in question for keyword in ('论文', '文献', '综述', '文章', '摘要', '总结')):
            return 'paper_summary'
        if any(keyword in question for keyword in ('表现', '特征', '征象', '影像')):
            return 'lesion_pattern'
        return 'mixed'

    def _suggest_follow_up_questions(self, question: str) -> list[str]:
        if any(keyword in question for keyword in ('鉴别', '区分', '区别', '对比')):
            return ['你可以继续追问这两类肿瘤在增强、弥散或灌注 MRI 上的差异。']
        if '序列' in question or any(keyword in question.lower() for keyword in ('dwi', 'adc', 'flair', 'swi', 't1', 't2')):
            return ['你可以继续追问该序列在高级别胶瘤、转移瘤或脑膜瘤中的典型意义。']
        return ['你可以继续追问某一种肿瘤、某个 MRI 序列，或要求比较两种肿瘤的影像鉴别。']

    def _extract_output_text(self, response: Any) -> str:
        output_text = getattr(response, 'output_text', None)
        if output_text:
            return str(output_text)
        return self._join_output_text(getattr(response, 'output', None))

    def _join_output_text(self, output_items: object) -> str:
        if not isinstance(output_items, Iterable):
            return ''

        chunks: list[str] = []
        for item in output_items:
            if getattr(item, 'type', None) != 'message':
                continue
            for content in getattr(item, 'content', []) or []:
                if getattr(content, 'type', None) == 'output_text':
                    chunks.append(getattr(content, 'text', ''))
        return ''.join(chunks).strip()

    def _extract_json_object(self, text: str) -> dict[str, Any] | list[Any] | None:
        if not text:
            return None
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in '{[':
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
                return payload
            except json.JSONDecodeError:
                continue
        return None


# 保留给测试或外部工具直接校验结构化 JSON 的入口。
def parse_brain_tumor_response(payload: str) -> BrainTumorQaResponse:
    return BrainTumorQaResponse.model_validate_json(payload)


# 将 evidence 模型列表转为普通字典，方便前端或测试直接消费。
def evidence_to_dicts(evidence: list[EvidenceReference]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in evidence]