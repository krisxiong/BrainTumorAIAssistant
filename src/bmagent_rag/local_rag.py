from __future__ import annotations
# 本文件负责本地 RAG 的文档解析、chunk 切分、BM25 索引构建和检索。
# sync.py 在建库时会调用 build_local_index()/save_local_index()，qa_service.py 在问答时会调用 load_local_index()/search_local_index()。

import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.html', '.htm'}
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
PROSE_CHAR_PATTERN = re.compile(r'[A-Za-z\u4e00-\u9fff]')
XML_METADATA_PATTERN = re.compile(r'</?(?:rdf|dc|xmp|pdf):', re.IGNORECASE)
REFERENCE_ENTRY_PATTERN = re.compile(r'(?:^|\\n)\\s*\\d+\\.\\s+[^\\n]{10,180}\\(\\d{4}\\)', re.MULTILINE)
DOI_PATTERN = re.compile(r'10\\.\\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
NOISE_MARKERS = (
    '%pdf-',
    'xref',
    'endobj',
    'endstream',
    '/linearized',
    '/creator(',
    '/producer(',
    'rdf:bag',
    'dc:format',
    'dc:identifier',
    'pdf:producer',
    'pdf:keywords',
    'crossmarkdomains',
)


class ParserDependencyMissing(RuntimeError):
    # 可选解析依赖缺失时抛出的异常，供 read_text_from_path() 区分“依赖缺失”和“文档损坏”。
    pass


class ParserExtractionFailed(RuntimeError):
    # 解析库存在但抽取失败时抛出的异常，供 manifest 标注失败原因。
    pass


# 单个原始文档的元数据对象，由 sync.py 扫描源目录后生成。
@dataclass(slots=True)
class LocalDocument:
    absolute_path: Path
    relative_path: str
    size_bytes: int
    sha256: str
    mime_type: str | None


# 可检索的 chunk 对象，保存原文、词频和文档归属信息。
@dataclass(slots=True)
class LocalChunk:
    chunk_id: str
    document_relative_path: str
    document_sha256: str
    chunk_index: int
    text: str
    token_count: int
    character_count: int
    term_frequencies: dict[str, int]


# 文档级摘要，供 manifest 、状态接口和前端展示使用。
@dataclass(slots=True)
class LocalDocumentSummary:
    relative_path: str
    absolute_path: str
    size_bytes: int
    sha256: str
    mime_type: str | None
    parser_name: str | None
    extracted_char_count: int
    chunk_count: int
    token_count: int


# 本地知识库索引对象，包含 chunks 和 BM25 所需的统计量。
@dataclass(slots=True)
class LocalKnowledgeBaseIndex:
    knowledge_base_id: str
    knowledge_base_name: str
    source_root: str
    index_path: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    created_at: datetime
    updated_at: datetime
    document_summaries: list[LocalDocumentSummary]
    chunks: list[LocalChunk]
    document_frequency: dict[str, int]
    average_chunk_length: float
    total_chunk_count: int
    total_token_count: int

    # 将索引序列化为 JSON 负载，供 local_index.json 落盘使用。
    def to_payload(self) -> dict[str, Any]:
        return {
            'knowledge_base_id': self.knowledge_base_id,
            'knowledge_base_name': self.knowledge_base_name,
            'source_root': self.source_root,
            'index_path': self.index_path,
            'chunk_size_chars': self.chunk_size_chars,
            'chunk_overlap_chars': self.chunk_overlap_chars,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'document_summaries': [asdict(item) for item in self.document_summaries],
            'chunks': [asdict(item) for item in self.chunks],
            'document_frequency': self.document_frequency,
            'average_chunk_length': self.average_chunk_length,
            'total_chunk_count': self.total_chunk_count,
            'total_token_count': self.total_token_count,
        }

    # 将 JSON 负载还原为 LocalKnowledgeBaseIndex 对象。
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> 'LocalKnowledgeBaseIndex':
        return cls(
            knowledge_base_id=str(payload.get('knowledge_base_id') or payload.get('vector_store_id') or ''),
            knowledge_base_name=str(payload.get('knowledge_base_name') or payload.get('vector_store_name') or 'brain-tumor-mri-kb-local'),
            source_root=str(payload.get('source_root') or ''),
            index_path=str(payload.get('index_path') or ''),
            chunk_size_chars=int(payload.get('chunk_size_chars') or 1400),
            chunk_overlap_chars=int(payload.get('chunk_overlap_chars') or 250),
            created_at=_parse_datetime(payload.get('created_at')),
            updated_at=_parse_datetime(payload.get('updated_at')),
            document_summaries=[LocalDocumentSummary(**item) for item in payload.get('document_summaries', [])],
            chunks=[LocalChunk(**item) for item in payload.get('chunks', [])],
            document_frequency={str(key): int(value) for key, value in (payload.get('document_frequency') or {}).items()},
            average_chunk_length=float(payload.get('average_chunk_length') or 0.0),
            total_chunk_count=int(payload.get('total_chunk_count') or 0),
            total_token_count=int(payload.get('total_token_count') or 0),
        )


# 一次检索命中的标准表示，qa_service.py 会基于它构造 prompt 和证据展示。
@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    file_name: str
    chunk_index: int
    score: float
    snippet: str
    token_count: int
    character_count: int
    document_relative_path: str


LocalSearchResult = SearchHit


# 基于知识库名称和源目录生成稳定 ID。
def build_knowledge_base_id(name: str, source_root: Path | str) -> str:
    seed = f'{Path(source_root).resolve()}::{name}'
    return f'local-bm25://{uuid5(NAMESPACE_URL, seed)}'


# 将文本分解为统一 token 列表，建库和检索阶段都会用到。
def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


# 统一换行、空白和 HTML entity，降低脏数据影响。
def normalize_text(text: str) -> str:
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    normalized = html.unescape(normalized)
    normalized = re.sub(r'[ \t\f\v]+', ' ', normalized)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    return normalized.strip()


# 判断文本是否更像 PDF 元数据、XML 元数据或二进制噪声，而不是自然语言正文。
def looks_like_noise_text(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True

    lowered = normalized.lower()
    control_char_count = len(CONTROL_CHAR_PATTERN.findall(normalized))
    printable_ratio = sum(1 for ch in normalized if ch.isprintable() or ch in '\n\t') / max(len(normalized), 1)
    prose_ratio = len(PROSE_CHAR_PATTERN.findall(normalized)) / max(len(normalized), 1)
    xml_tag_count = len(XML_METADATA_PATTERN.findall(normalized))
    marker_hits = sum(marker in lowered for marker in NOISE_MARKERS)
    reference_entry_count = len(REFERENCE_ENTRY_PATTERN.findall(normalized))
    doi_count = len(DOI_PATTERN.findall(normalized))
    et_al_count = lowered.count('et al.')

    if lowered.startswith('%pdf-'):
        return True
    if control_char_count >= 3 and prose_ratio < 0.30:
        return True
    if printable_ratio < 0.92 and prose_ratio < 0.35:
        return True
    if xml_tag_count >= 4:
        return True
    if marker_hits >= 4 and prose_ratio < 0.55:
        return True
    if ('references' in lowered or 'bibliography' in lowered) and reference_entry_count >= 2:
        return True
    if reference_entry_count >= 3 and (doi_count >= 2 or et_al_count >= 2):
        return True
    if len(normalized) >= 180 and prose_ratio < 0.12:
        return True
    return False


# 按固定字符窗口切 chunk，是本地 RAG 召回的基础。
def split_into_chunks(text: str, chunk_size_chars: int, chunk_overlap_chars: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    chunk_size = max(int(chunk_size_chars), 1)
    overlap = max(int(chunk_overlap_chars), 0)
    if overlap >= chunk_size:
        overlap = max(chunk_size - 1, 0)
    step = max(chunk_size - overlap, 1)

    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start += step
    return chunks


# 根据文件类型选择解析方式。PDF / DOCX 缺依赖时不再强行按二进制解码入库，避免污染索引。
def read_text_from_path(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md', '.csv', '.json'}:
        return _read_text_file(path), 'plain_text'
    if suffix in {'.html', '.htm'}:
        raw = _read_text_file(path)
        raw = re.sub(r'<script.*?</script>', ' ', raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r'<style.*?</style>', ' ', raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r'<[^>]+>', ' ', raw)
        return normalize_text(raw), 'html'
    if suffix == '.pdf':
        try:
            return _extract_pdf_text(path), 'pdf'
        except ParserDependencyMissing:
            return '', 'pdf_missing_dependency'
        except ParserExtractionFailed:
            return '', 'pdf_parse_failed'
    if suffix == '.docx':
        try:
            return _extract_docx_text(path), 'docx'
        except ParserDependencyMissing:
            return '', 'docx_missing_dependency'
        except ParserExtractionFailed:
            return '', 'docx_parse_failed'
    return _read_text_file(path), 'plain_text'


# 以多编码策略读取文本文件，提高兼容性。
def _read_text_file(path: Path) -> str:
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030', 'cp936', 'latin-1'):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding='utf-8', errors='ignore')


# 使用 pypdf 从 PDF 中抽取文本。若依赖缺失或抽取失败，会由上层记录成失败状态，而不是把二进制垃圾写入索引。
def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise ParserDependencyMissing('pypdf is required to parse PDF files.') from exc

    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ''
            if extracted:
                pages.append(extracted)
        return normalize_text('\n\n'.join(pages))
    except Exception as exc:
        raise ParserExtractionFailed(f'failed to extract PDF text: {path.name}') from exc


# 使用 python-docx 从 DOCX 中抽取文本。和 PDF 一样，失败时不再回退为二进制乱码。
def _extract_docx_text(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception as exc:
        raise ParserDependencyMissing('python-docx is required to parse DOCX files.') from exc

    try:
        document = Document(str(path))
        return normalize_text('\n\n'.join(paragraph.text for paragraph in document.paragraphs if paragraph.text))
    except Exception as exc:
        raise ParserExtractionFailed(f'failed to extract DOCX text: {path.name}') from exc


# 对普通未知文本做最后兜底时仍可用，但不会再用于 PDF / DOCX。
def _binary_fallback_text(path: Path) -> str:
    return normalize_text(path.read_bytes().decode('utf-8', errors='ignore'))


# 建库主函数：解析文档、切块并生成 BM25 所需的统计结构。
def build_local_index(
    documents: Iterable[LocalDocument],
    *,
    knowledge_base_name: str,
    source_root: Path | str,
    index_path: Path | str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    knowledge_base_id: str | None = None,
) -> LocalKnowledgeBaseIndex:
    resolved_id = knowledge_base_id or build_knowledge_base_id(knowledge_base_name, source_root)
    created_at = datetime.now(timezone.utc)
    document_summaries: list[LocalDocumentSummary] = []
    chunks: list[LocalChunk] = []
    document_frequency: Counter[str] = Counter()
    total_token_count = 0

    for document in documents:
        text, parser_name = read_text_from_path(document.absolute_path)
        normalized = normalize_text(text)
        chunk_texts = split_into_chunks(normalized, chunk_size_chars, chunk_overlap_chars)
        token_count_for_document = 0
        effective_chunk_count = 0
        parser_name_for_summary = parser_name

        for chunk_index, chunk_text in enumerate(chunk_texts):
            if looks_like_noise_text(chunk_text):
                continue
            tokens = tokenize(chunk_text)
            if not tokens:
                continue
            term_frequencies = Counter(tokens)
            document_frequency.update(set(tokens))
            token_count = len(tokens)
            total_token_count += token_count
            token_count_for_document += token_count
            effective_chunk_count += 1
            chunks.append(
                LocalChunk(
                    chunk_id=f'{document.relative_path}::{chunk_index:04d}',
                    document_relative_path=document.relative_path,
                    document_sha256=document.sha256,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=token_count,
                    character_count=len(chunk_text),
                    term_frequencies=dict(term_frequencies),
                )
            )

        if normalized and effective_chunk_count == 0 and parser_name in {'pdf', 'docx'}:
            parser_name_for_summary = f'{parser_name}_filtered_as_noise'

        document_summaries.append(
            LocalDocumentSummary(
                relative_path=document.relative_path,
                absolute_path=str(document.absolute_path),
                size_bytes=document.size_bytes,
                sha256=document.sha256,
                mime_type=document.mime_type,
                parser_name=parser_name_for_summary,
                extracted_char_count=len(normalized),
                chunk_count=effective_chunk_count,
                token_count=token_count_for_document,
            )
        )

    average_chunk_length = (total_token_count / len(chunks)) if chunks else 0.0
    return LocalKnowledgeBaseIndex(
        knowledge_base_id=resolved_id,
        knowledge_base_name=knowledge_base_name,
        source_root=str(Path(source_root)),
        index_path=str(Path(index_path)),
        chunk_size_chars=int(chunk_size_chars),
        chunk_overlap_chars=int(chunk_overlap_chars),
        created_at=created_at,
        updated_at=created_at,
        document_summaries=document_summaries,
        chunks=chunks,
        document_frequency=dict(document_frequency),
        average_chunk_length=average_chunk_length,
        total_chunk_count=len(chunks),
        total_token_count=total_token_count,
    )


# 将索引安全写入 local_index.json，供后续问答流程加载。
def save_local_index(index: LocalKnowledgeBaseIndex, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.updated_at = datetime.now(timezone.utc)
    payload = index.to_payload()
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(path)


# 从 local_index.json 读取本地索引。
def load_local_index(path: Path | str) -> LocalKnowledgeBaseIndex:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return LocalKnowledgeBaseIndex.from_payload(payload)


# 对本地索引执行 BM25 检索。即使索引是旧版本，也会在查询时再过滤一遍噪声 chunk，避免脏片段继续命中前排。
def search_local_index(
    index: LocalKnowledgeBaseIndex,
    query: str,
    *,
    top_k: int = 5,
    max_snippet_chars: int = 520,
) -> list[SearchHit]:
    query_tokens = tokenize(normalize_text(query))
    if not query_tokens or not index.chunks:
        return []

    scores: list[tuple[float, LocalChunk]] = []
    total_chunks = len(index.chunks)
    avgdl = index.average_chunk_length or 1.0
    query_term_counts = Counter(query_tokens)

    for chunk in index.chunks:
        if looks_like_noise_text(chunk.text):
            continue
        score = 0.0
        for term, qtf in query_term_counts.items():
            tf = chunk.term_frequencies.get(term)
            if not tf:
                continue
            df = index.document_frequency.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + ((total_chunks - df + 0.5) / (df + 0.5)))
            numerator = tf * 2.2
            denominator = tf + 1.2 * (1 - 0.75 + 0.75 * (chunk.token_count / avgdl))
            score += idf * numerator / denominator * qtf
        if score > 0:
            scores.append((score, chunk))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [
        SearchHit(
            chunk_id=chunk.chunk_id,
            file_name=chunk.document_relative_path,
            chunk_index=chunk.chunk_index,
            score=round(score, 6),
            snippet=_truncate_snippet(chunk.text, max_snippet_chars),
            token_count=chunk.token_count,
            character_count=chunk.character_count,
            document_relative_path=chunk.document_relative_path,
        )
        for score, chunk in scores[:top_k]
    ]


# 截断 chunk 文本，便于前端展示和 prompt 拼接。
def _truncate_snippet(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max(0, max_chars - 1)].rstrip() + '...'


# 将输入值解析为 datetime 对象，保持元数据结构一致。
def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)