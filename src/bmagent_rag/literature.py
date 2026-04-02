# 本文件负责安全版文献采集：先用 PubMed 拿候选元数据，再用 OpenAlex 补充开放获取信息，最后按配置决定是否下载 OA PDF。
# sync.py 会把下载后的文献进一步入库，所以这个文件是“文献获取链路”的入口。

from __future__ import annotations

import argparse
import csv
import json
import re
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .config import load_env_file


NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_OUTPUT_DIR = Path("data") / "literature_candidates"
DEFAULT_DOWNLOAD_DIR = Path("data") / "knowledge_base" / "source" / "papers"


@dataclass(slots=True)
# 文献候选采集所需的查询条件、输出目录和 API 参数配置。
class LiteratureSearchConfig:
    query: str
    max_results: int = 20
    from_year: int | None = None
    to_year: int | None = None
    reviews_only: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    download_open_access: bool = False
    download_dir: Path = DEFAULT_DOWNLOAD_DIR
    contact_email: str | None = None
    tool_name: str = "bmagent-literature-search"
    ncbi_api_key: str | None = None
    timeout_seconds: float = 30.0
    ncbi_pause_seconds: float = 0.34
    openalex_pause_seconds: float = 0.11


@dataclass(slots=True)
# 一篇候选文献的标准数据结构，包含 PubMed 元数据和 OpenAlex 补充信息。
class LiteratureCandidate:
    rank: int
    source_query: str
    pmid: str
    title: str
    journal: str | None
    publication_date: str | None
    publication_year: int | None
    doi: str | None
    pmc_id: str | None
    authors: list[str]
    article_types: list[str]
    is_review: bool
    open_access_status: str
    oa_pdf_url: str | None
    oa_landing_page_url: str | None
    openalex_id: str | None
    cited_by_count: int | None
    downloaded_pdf_path: str | None = None

    # 处理 to_csv_row 相关逻辑，供文献候选采集与 OA 下载流程使用。
    def to_csv_row(self) -> dict[str, Any]:
        data = asdict(self)
        data["authors"] = "; ".join(self.authors)
        data["article_types"] = "; ".join(self.article_types)
        return data


@dataclass(slots=True)
# 一次文献采集的执行结果，包含候选列表、导出文件和下载统计。
class LiteratureSearchResult:
    query: str
    generated_at: str
    csv_path: Path
    json_path: Path
    total_candidates: int
    open_access_candidates: int
    downloaded_pdfs: int
    candidates: list[LiteratureCandidate]


# 处理 _build_headers 相关逻辑，供文献候选采集与 OA 下载流程使用。
def _build_headers(contact_email: str | None, tool_name: str) -> dict[str, str]:
    agent = tool_name
    if contact_email:
        agent = f"{tool_name} ({contact_email})"
    return {
        "User-Agent": agent,
        "Accept": "application/json",
    }


# 处理 _get_json 相关逻辑，供文献候选采集与 OA 下载流程使用。
def _get_json(url: str, *, headers: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


# 处理 _download_binary 相关逻辑，供文献候选采集与 OA 下载流程使用。
def _download_binary(url: str, *, headers: dict[str, str], timeout_seconds: float) -> bytes:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


# 仅接受真正的 PDF 文件，避免把 HTML 落地页误存成 .pdf。
def _looks_like_pdf_binary(binary: bytes) -> bool:
    header = binary.lstrip()[:5]
    return header.startswith(b'%PDF-')


# 处理 build_pubmed_esearch_url 相关逻辑，供文献候选采集与 OA 下载流程使用。
def build_pubmed_esearch_url(config: LiteratureSearchConfig) -> str:
    query = config.query.strip()
    if config.reviews_only:
        query = f"({query}) AND review[Publication Type]"

    params = {
        "db": "pubmed",
        "retmode": "json",
        "sort": "relevance",
        "retmax": str(config.max_results),
        "term": query,
        "tool": config.tool_name,
    }
    if config.contact_email:
        params["email"] = config.contact_email
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    if config.from_year is not None or config.to_year is not None:
        params["datetype"] = "pdat"
        if config.from_year is not None:
            params["mindate"] = str(config.from_year)
        if config.to_year is not None:
            params["maxdate"] = str(config.to_year)
    return f"{NCBI_ESEARCH_URL}?{urlencode(params)}"


# 处理 build_pubmed_esummary_url 相关逻辑，供文献候选采集与 OA 下载流程使用。
def build_pubmed_esummary_url(pmids: Sequence[str], config: LiteratureSearchConfig) -> str:
    params = {
        "db": "pubmed",
        "retmode": "json",
        "id": ",".join(pmids),
        "tool": config.tool_name,
    }
    if config.contact_email:
        params["email"] = config.contact_email
    if config.ncbi_api_key:
        params["api_key"] = config.ncbi_api_key
    return f"{NCBI_ESUMMARY_URL}?{urlencode(params)}"


# 处理 build_openalex_work_url 相关逻辑，供文献候选采集与 OA 下载流程使用。
def build_openalex_work_url(candidate: LiteratureCandidate, config: LiteratureSearchConfig) -> str | None:
    params: dict[str, str] = {}
    if config.contact_email:
        params["mailto"] = config.contact_email

    identifier: str | None = None
    if candidate.doi:
        identifier = f"https://doi.org/{candidate.doi}"
    elif candidate.pmid:
        identifier = f"https://pubmed.ncbi.nlm.nih.gov/{candidate.pmid}"
    if identifier is None:
        return None

    encoded_identifier = quote(identifier, safe="")
    return f"{OPENALEX_WORKS_URL}/{encoded_identifier}?{urlencode(params)}" if params else f"{OPENALEX_WORKS_URL}/{encoded_identifier}"


# 向 PubMed esearch 请求 PMID 列表，作为候选采集的第一步。
def search_pubmed_pmids(
    config: LiteratureSearchConfig,
    fetch_json: Callable[..., dict[str, Any]] = _get_json,
) -> list[str]:
    url = build_pubmed_esearch_url(config)
    payload = fetch_json(
        url,
        headers=_build_headers(config.contact_email, config.tool_name),
        timeout_seconds=config.timeout_seconds,
    )
    idlist = payload.get("esearchresult", {}).get("idlist", [])
    return [str(item) for item in idlist]


# 根据 PMID 批量获取 PubMed summary 元数据。
def fetch_pubmed_summaries(
    pmids: Sequence[str],
    config: LiteratureSearchConfig,
    fetch_json: Callable[..., dict[str, Any]] = _get_json,
) -> list[dict[str, Any]]:
    if not pmids:
        return []

    url = build_pubmed_esummary_url(pmids, config)
    payload = fetch_json(
        url,
        headers=_build_headers(config.contact_email, config.tool_name),
        timeout_seconds=config.timeout_seconds,
    )
    result = payload.get("result", {})
    summaries: list[dict[str, Any]] = []
    for pmid in pmids:
        record = result.get(str(pmid))
        if record:
            summaries.append(record)
    return summaries


# 处理 normalize_pubmed_candidate 相关逻辑，供文献候选采集与 OA 下载流程使用。
def normalize_pubmed_candidate(summary: dict[str, Any], rank: int, source_query: str) -> LiteratureCandidate:
    article_ids = summary.get("articleids", []) or []
    doi = _pick_article_id(article_ids, "doi")
    pmc_id = _pick_article_id(article_ids, "pmc") or _pick_article_id(article_ids, "pmcid")
    pubdate = summary.get("pubdate") or summary.get("sortpubdate")
    article_types = [str(item) for item in summary.get("pubtype", []) or []]
    authors = [item.get("name") for item in summary.get("authors", []) if item.get("name")]
    publication_year = _extract_year(pubdate)

    return LiteratureCandidate(
        rank=rank,
        source_query=source_query,
        pmid=str(summary.get("uid") or ""),
        title=str(summary.get("title") or "").strip(),
        journal=(summary.get("fulljournalname") or summary.get("source") or None),
        publication_date=pubdate,
        publication_year=publication_year,
        doi=doi,
        pmc_id=pmc_id,
        authors=authors,
        article_types=article_types,
        is_review=any("review" in item.lower() for item in article_types),
        open_access_status="unknown",
        oa_pdf_url=None,
        oa_landing_page_url=None,
        openalex_id=None,
        cited_by_count=None,
    )


# 用 OpenAlex 补充开放获取状态、PDF 链接和引用次数。
def enrich_with_openalex(
    candidate: LiteratureCandidate,
    config: LiteratureSearchConfig,
    fetch_json: Callable[..., dict[str, Any]] = _get_json,
) -> LiteratureCandidate:
    url = build_openalex_work_url(candidate, config)
    if not url:
        return candidate

    try:
        payload = fetch_json(
            url,
            headers=_build_headers(config.contact_email, config.tool_name),
            timeout_seconds=config.timeout_seconds,
        )
    except HTTPError as exc:
        if exc.code == 404:
            return candidate
        raise
    except URLError:
        return candidate

    open_access = payload.get("open_access") or {}
    best_oa_location = payload.get("best_oa_location") or {}
    primary_location = payload.get("primary_location") or {}
    pdf_url = (
        best_oa_location.get("pdf_url")
        or primary_location.get("pdf_url")
        or (primary_location.get("pdf_url") if isinstance(primary_location, dict) else None)
    )
    landing_page_url = (
        best_oa_location.get("landing_page_url")
        or primary_location.get("landing_page_url")
        or open_access.get("oa_url")
    )
    oa_status = "open" if open_access.get("is_oa") else "closed"
    if open_access == {}:
        oa_status = "unknown"

    candidate.open_access_status = oa_status
    candidate.oa_pdf_url = pdf_url
    candidate.oa_landing_page_url = landing_page_url
    candidate.openalex_id = payload.get("id")
    candidate.cited_by_count = payload.get("cited_by_count")
    return candidate


# 处理 export_candidates 相关逻辑，供文献候选采集与 OA 下载流程使用。
def export_candidates(candidates: Sequence[LiteratureCandidate], output_dir: Path, query: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = slugify(query)[:80] or "literature_search"
    csv_path = output_dir / f"{stamp}_{slug}.csv"
    json_path = output_dir / f"{stamp}_{slug}.json"

    rows = [candidate.to_csv_row() for candidate in candidates]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else list(LiteratureCandidate.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "query": query,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


# 仅在开启配置时下载 OpenAlex 明确给出的 OA PDF。
def maybe_download_open_access_pdfs(
    candidates: Sequence[LiteratureCandidate],
    config: LiteratureSearchConfig,
    download_binary: Callable[..., bytes] = _download_binary,
) -> int:
    # 只有显式开启下载时才会真正拉取 PDF，默认保持“检索候选但不自动下载”。
    if not config.download_open_access:
        return 0

    config.download_dir.mkdir(parents=True, exist_ok=True)
    # 请求头会带上工具名和联系人信息，尽量符合上游 API 的礼貌访问要求。
    headers = _build_headers(config.contact_email, config.tool_name)
    downloaded = 0

    for candidate in candidates:
        # 只下载 OpenAlex 已明确给出 PDF 直链的开放获取论文。
        if not candidate.oa_pdf_url:
            continue
        try:
            binary = download_binary(
                candidate.oa_pdf_url,
                headers=headers,
                timeout_seconds=config.timeout_seconds,
            )
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue

        if not _looks_like_pdf_binary(binary):
            continue

        target = config.download_dir / build_candidate_filename(candidate, url=candidate.oa_pdf_url)
        target.write_bytes(binary)
        candidate.downloaded_pdf_path = str(target)
        downloaded += 1
        time.sleep(config.openalex_pause_seconds)

    return downloaded


# 文献采集主流程：PubMed 检索 -> OpenAlex 补充 -> 可选下载 -> 导出候选清单。
def collect_literature_candidates(
    config: LiteratureSearchConfig,
    fetch_json: Callable[..., dict[str, Any]] = _get_json,
    download_binary: Callable[..., bytes] = _download_binary,
) -> LiteratureSearchResult:
    # 先从 PubMed 拿 PMID 列表，再根据 PMID 获取每篇文献的 summary 元数据。
    pmids = search_pubmed_pmids(config, fetch_json=fetch_json)
    time.sleep(config.ncbi_pause_seconds)
    summaries = fetch_pubmed_summaries(pmids, config, fetch_json=fetch_json)

    # 先把 PubMed summary 归一化为项目内部统一的 LiteratureCandidate 对象。
    candidates: list[LiteratureCandidate] = []
    for index, summary in enumerate(summaries, start=1):
        candidate = normalize_pubmed_candidate(summary, rank=index, source_query=config.query)
        candidates.append(candidate)

    for candidate in candidates:
        # 再用 OpenAlex 补充开放获取状态、PDF 链接和引用次数等额外信息。
        enrich_with_openalex(candidate, config, fetch_json=fetch_json)
        time.sleep(config.openalex_pause_seconds)

    # 最后按配置决定是否下载 OA PDF，并始终导出 CSV/JSON 候选清单供人工确认。
    downloaded = maybe_download_open_access_pdfs(candidates, config, download_binary=download_binary)
    csv_path, json_path = export_candidates(candidates, config.output_dir, config.query)

    return LiteratureSearchResult(
        query=config.query,
        generated_at=datetime.now(timezone.utc).isoformat(),
        csv_path=csv_path,
        json_path=json_path,
        total_candidates=len(candidates),
        open_access_candidates=sum(1 for item in candidates if item.open_access_status == "open"),
        downloaded_pdfs=downloaded,
        candidates=candidates,
    )


# 处理 build_parser 相关逻辑，供文献候选采集与 OA 下载流程使用。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search PubMed candidates and enrich them with OpenAlex open-access metadata."
    )
    parser.add_argument("query", help="Literature query, for example: glioblastoma MRI review")
    parser.add_argument("--max-results", type=int, default=20, help="Number of PubMed candidates to collect.")
    parser.add_argument("--from-year", type=int, default=None, help="Optional lower year bound.")
    parser.add_argument("--to-year", type=int, default=None, help="Optional upper year bound.")
    parser.add_argument("--reviews-only", action="store_true", help="Restrict PubMed search to review articles.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write CSV/JSON candidate lists.")
    parser.add_argument("--download-open-access", action="store_true", help="Download only explicit open-access PDFs into the knowledge base folder.")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="Where OA PDFs should be saved if enabled.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env file to load first.")
    parser.add_argument("--contact-email", default=None, help="Contact email sent to NCBI/OpenAlex when available.")
    parser.add_argument("--tool-name", default="bmagent-literature-search", help="Tool name sent to upstream APIs.")
    parser.add_argument("--ncbi-api-key", default=None, help="Optional NCBI API key.")
    return parser


# 处理 build_config_from_args 相关逻辑，供文献候选采集与 OA 下载流程使用。
def build_config_from_args(args: argparse.Namespace) -> LiteratureSearchConfig:
    return LiteratureSearchConfig(
        query=args.query,
        max_results=args.max_results,
        from_year=args.from_year,
        to_year=args.to_year,
        reviews_only=args.reviews_only,
        output_dir=args.output_dir,
        download_open_access=args.download_open_access,
        download_dir=args.download_dir,
        contact_email=args.contact_email,
        tool_name=args.tool_name,
        ncbi_api_key=args.ncbi_api_key,
    )


# 命令行入口，由 scripts/search_literature_candidates.py 调用。
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.env_file:
        load_env_file(args.env_file)

    if not args.contact_email:
        args.contact_email = _read_env("BMAGENT_CONTACT_EMAIL") or _read_env("OPENALEX_MAILTO")
    if not args.ncbi_api_key:
        args.ncbi_api_key = _read_env("NCBI_API_KEY")
    if args.tool_name == "bmagent-literature-search":
        args.tool_name = _read_env("BMAGENT_TOOL_NAME") or args.tool_name

    config = build_config_from_args(args)
    result = collect_literature_candidates(config)

    print(f"Query: {result.query}")
    print(f"Candidates: {result.total_candidates}")
    print(f"Open access candidates: {result.open_access_candidates}")
    print(f"Downloaded PDFs: {result.downloaded_pdfs}")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    return 0


# 处理 slugify 相关逻辑，供文献候选采集与 OA 下载流程使用。
def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


# 处理 build_candidate_filename 相关逻辑，供文献候选采集与 OA 下载流程使用。
def build_candidate_filename(candidate: LiteratureCandidate, url: str) -> str:
    suffix = Path(urlparse(url).path).suffix or ".pdf"
    title_slug = slugify(candidate.title)[:80] or f"pmid_{candidate.pmid}"
    return f"{candidate.rank:02d}_{title_slug}{suffix}"


# 处理 _pick_article_id 相关逻辑，供文献候选采集与 OA 下载流程使用。
def _pick_article_id(article_ids: Sequence[dict[str, Any]], id_type: str) -> str | None:
    target = id_type.lower()
    for item in article_ids:
        current = str(item.get("idtype") or "").lower()
        if current == target:
            value = item.get("value")
            return str(value) if value else None
    return None


# 处理 _extract_year 相关逻辑，供文献候选采集与 OA 下载流程使用。
def _extract_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    if not match:
        return None
    return int(match.group(0))


# 处理 _read_env 相关逻辑，供文献候选采集与 OA 下载流程使用。
def _read_env(name: str) -> str | None:
    import os

    value = os.getenv(name)
    return value or None
