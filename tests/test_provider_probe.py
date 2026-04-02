from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bmagent_rag.provider_probe import ProviderProbeConfig, run_provider_probe


class FakeResponses:
    def __init__(self, marker_holder: dict[str, str], should_fail: bool = False) -> None:
        self.marker_holder = marker_holder
        self.should_fail = should_fail

    def create(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("responses unsupported")

        tools = kwargs.get("tools") or []
        if tools:
            marker = self.marker_holder["marker"]
            result = SimpleNamespace(file_id="file_123", filename="probe.txt", text=f"Probe token: {marker}", score=0.99)
            file_search_call = SimpleNamespace(type="file_search_call", results=[result])
            message = SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=marker)],
            )
            return SimpleNamespace(id="resp_rag", output_text=marker, output=[file_search_call, message])

        return SimpleNamespace(id="resp_basic", output_text="OK", output=[])


class FakeFiles:
    def create(self, **kwargs):
        content = kwargs["file"].read().decode("utf-8")
        marker = content.strip().splitlines()[-1].split(": ", 1)[-1]
        self.marker_holder["marker"] = marker
        return SimpleNamespace(id="file_123", status="uploaded")

    def __init__(self, marker_holder: dict[str, str]) -> None:
        self.marker_holder = marker_holder
        self.deleted = []

    def delete(self, file_id: str):
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)


class FakeVectorStoreFiles:
    def __init__(self) -> None:
        self.retrieve_count = 0

    def create(self, **kwargs):
        return SimpleNamespace(id="vsf_123", status="in_progress")

    def retrieve(self, **kwargs):
        self.retrieve_count += 1
        status = "completed" if self.retrieve_count >= 1 else "in_progress"
        return SimpleNamespace(id="vsf_123", status=status)


class FakeVectorStores:
    def __init__(self) -> None:
        self.files = FakeVectorStoreFiles()
        self.deleted = []

    def create(self, **kwargs):
        return SimpleNamespace(id="vs_123")

    def delete(self, vector_store_id: str):
        self.deleted.append(vector_store_id)
        return SimpleNamespace(id=vector_store_id, deleted=True)


class FakeClient:
    def __init__(self, should_fail: bool = False) -> None:
        self.marker_holder: dict[str, str] = {}
        self.responses = FakeResponses(self.marker_holder, should_fail=should_fail)
        self.files = FakeFiles(self.marker_holder)
        self.vector_stores = FakeVectorStores()


def test_provider_probe_success(tmp_path: Path) -> None:
    client = FakeClient()
    config = ProviderProbeConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="demo-model",
        poll_interval_seconds=0,
        poll_timeout_seconds=1,
        report_dir=tmp_path,
    )

    report = run_provider_probe(config, client=client)

    assert report.overall_ok is True
    assert report.file_id == "file_123"
    assert report.vector_store_id == "vs_123"
    assert any(step.name == "responses_file_search" and step.ok for step in report.steps)
    assert Path(report.report_path).exists()
    payload = json.loads(Path(report.report_path).read_text(encoding="utf-8"))
    assert payload["overall_ok"] is True


def test_provider_probe_stops_when_responses_fail(tmp_path: Path) -> None:
    client = FakeClient(should_fail=True)
    config = ProviderProbeConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="demo-model",
        report_dir=tmp_path,
    )

    report = run_provider_probe(config, client=client)

    assert report.overall_ok is False
    assert report.steps[0].name == "responses_basic"
    assert report.steps[0].ok is False