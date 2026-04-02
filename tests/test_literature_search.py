from __future__ import annotations

import csv
import json
from pathlib import Path

from bmagent_rag.literature import (
    LiteratureSearchConfig,
    build_candidate_filename,
    build_pubmed_esearch_url,
    collect_literature_candidates,
    normalize_pubmed_candidate,
)


def test_build_pubmed_esearch_url_applies_review_and_date_filters() -> None:
    config = LiteratureSearchConfig(
        query="glioblastoma MRI",
        max_results=15,
        from_year=2020,
        to_year=2024,
        reviews_only=True,
        contact_email="demo@example.com",
    )

    url = build_pubmed_esearch_url(config)

    assert "retmax=15" in url
    assert "mindate=2020" in url
    assert "maxdate=2024" in url
    assert "review%5BPublication+Type%5D" in url
    assert "email=demo%40example.com" in url


def test_normalize_pubmed_candidate_extracts_ids_and_review_flag() -> None:
    summary = {
        "uid": "12345678",
        "title": "MRI review of glioblastoma",
        "fulljournalname": "Neuro Oncology Review",
        "pubdate": "2024 Jan",
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "pubtype": ["Review", "Journal Article"],
        "articleids": [
            {"idtype": "doi", "value": "10.1000/demo"},
            {"idtype": "pmc", "value": "PMC123456"},
        ],
    }

    candidate = normalize_pubmed_candidate(summary, rank=1, source_query="glioblastoma MRI")

    assert candidate.pmid == "12345678"
    assert candidate.doi == "10.1000/demo"
    assert candidate.pmc_id == "PMC123456"
    assert candidate.is_review is True
    assert candidate.publication_year == 2024


def test_collect_literature_candidates_exports_files_and_downloads_oa_pdf(tmp_path: Path) -> None:
    responses = {
        "esearch": {"esearchresult": {"idlist": ["12345678"]}},
        "esummary": {
            "result": {
                "uids": ["12345678"],
                "12345678": {
                    "uid": "12345678",
                    "title": "Advanced MRI in glioblastoma",
                    "fulljournalname": "Radiology Review",
                    "pubdate": "2023 Feb",
                    "authors": [{"name": "Alice"}],
                    "pubtype": ["Review"],
                    "articleids": [{"idtype": "doi", "value": "10.1000/demo"}],
                },
            }
        },
        "openalex": {
            "id": "https://openalex.org/W123",
            "cited_by_count": 42,
            "open_access": {"is_oa": True, "oa_url": "https://example.org/landing"},
            "best_oa_location": {
                "pdf_url": "https://example.org/paper.pdf",
                "landing_page_url": "https://example.org/landing",
            },
        },
    }

    def fake_fetch_json(url: str, **kwargs):
        if "esearch.fcgi" in url:
            return responses["esearch"]
        if "esummary.fcgi" in url:
            return responses["esummary"]
        if "api.openalex.org/works/" in url:
            return responses["openalex"]
        raise AssertionError(url)

    def fake_download_binary(url: str, **kwargs):
        assert url == "https://example.org/paper.pdf"
        return b"%PDF-demo"

    config = LiteratureSearchConfig(
        query="glioblastoma MRI review",
        max_results=5,
        output_dir=tmp_path / "outputs",
        download_open_access=True,
        download_dir=tmp_path / "downloads",
        ncbi_pause_seconds=0,
        openalex_pause_seconds=0,
    )

    result = collect_literature_candidates(
        config,
        fetch_json=fake_fetch_json,
        download_binary=fake_download_binary,
    )

    assert result.total_candidates == 1
    assert result.open_access_candidates == 1
    assert result.downloaded_pdfs == 1
    assert result.csv_path.exists()
    assert result.json_path.exists()

    csv_rows = list(csv.DictReader(result.csv_path.open("r", encoding="utf-8-sig")))
    assert csv_rows[0]["pmid"] == "12345678"
    assert csv_rows[0]["open_access_status"] == "open"

    json_payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert json_payload["candidates"][0]["openalex_id"] == "https://openalex.org/W123"

    downloaded_files = list((tmp_path / "downloads").glob("*.pdf"))
    assert len(downloaded_files) == 1
    assert downloaded_files[0].name == build_candidate_filename(result.candidates[0], url="https://example.org/paper.pdf")

def test_collect_literature_candidates_skips_html_disguised_as_pdf(tmp_path: Path) -> None:
    responses = {
        "esearch": {"esearchresult": {"idlist": ["12345678"]}},
        "esummary": {
            "result": {
                "uids": ["12345678"],
                "12345678": {
                    "uid": "12345678",
                    "title": "Advanced MRI in glioblastoma",
                    "fulljournalname": "Radiology Review",
                    "pubdate": "2023 Feb",
                    "authors": [{"name": "Alice"}],
                    "pubtype": ["Review"],
                    "articleids": [{"idtype": "doi", "value": "10.1000/demo"}],
                },
            }
        },
        "openalex": {
            "id": "https://openalex.org/W123",
            "cited_by_count": 42,
            "open_access": {"is_oa": True, "oa_url": "https://example.org/landing"},
            "best_oa_location": {
                "pdf_url": "https://example.org/not-really-a-pdf",
                "landing_page_url": "https://example.org/landing",
            },
        },
    }

    def fake_fetch_json(url: str, **kwargs):
        if "esearch.fcgi" in url:
            return responses["esearch"]
        if "esummary.fcgi" in url:
            return responses["esummary"]
        if "api.openalex.org/works/" in url:
            return responses["openalex"]
        raise AssertionError(url)

    def fake_download_binary(url: str, **kwargs):
        assert url == "https://example.org/not-really-a-pdf"
        return b"\n\n<html>not a pdf</html>"

    config = LiteratureSearchConfig(
        query="glioblastoma MRI review",
        max_results=5,
        output_dir=tmp_path / "outputs",
        download_open_access=True,
        download_dir=tmp_path / "downloads",
        ncbi_pause_seconds=0,
        openalex_pause_seconds=0,
    )

    result = collect_literature_candidates(
        config,
        fetch_json=fake_fetch_json,
        download_binary=fake_download_binary,
    )

    assert result.total_candidates == 1
    assert result.open_access_candidates == 1
    assert result.downloaded_pdfs == 0
    assert list((tmp_path / "downloads").glob("*.pdf")) == []
    assert result.candidates[0].downloaded_pdf_path is None