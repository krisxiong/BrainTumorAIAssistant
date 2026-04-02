from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence
from uuid import uuid4

from openai import OpenAI

from .config import load_env_file


TERMINAL_VECTOR_STORE_STATUSES = {"completed", "failed", "cancelled", "expired"}
DEFAULT_REPORT_DIR = Path("storage") / "provider_probe"


@dataclass(slots=True)
class ProviderProbeConfig:
    api_key: str
    base_url: str | None
    model: str
    timeout_seconds: float = 60.0
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 120.0
    keep_artifacts: bool = False
    report_dir: Path = DEFAULT_REPORT_DIR


@dataclass(slots=True)
class ProbeStepResult:
    name: str
    ok: bool
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderProbeReport:
    generated_at: str
    base_url: str | None
    model: str
    overall_ok: bool
    steps: list[ProbeStepResult]
    file_id: str | None = None
    vector_store_id: str | None = None
    report_path: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "generated_at": self.generated_at,
                "base_url": self.base_url,
                "model": self.model,
                "overall_ok": self.overall_ok,
                "steps": [asdict(step) for step in self.steps],
                "file_id": self.file_id,
                "vector_store_id": self.vector_store_id,
                "report_path": self.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe whether an OpenAI-compatible provider supports Responses, Files, vector_store, and file_search."
    )
    parser.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional .env file to load first.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--keep-artifacts", action="store_true", help="Do not attempt to delete the temporary uploaded file or vector store.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def build_config_from_args(args: argparse.Namespace) -> ProviderProbeConfig:
    return ProviderProbeConfig(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL") or None,
        model=args.model or os.getenv("OPENAI_MODEL", "gpt-5.4"),
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        poll_timeout_seconds=args.poll_timeout_seconds,
        keep_artifacts=args.keep_artifacts,
        report_dir=args.report_dir,
    )


def run_provider_probe(config: ProviderProbeConfig, client: OpenAI | None = None) -> ProviderProbeReport:
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY 未配置，无法执行探针。")

    report = ProviderProbeReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        base_url=config.base_url,
        model=config.model,
        overall_ok=False,
        steps=[],
    )

    client = client or OpenAI(
        api_key=config.api_key,
        base_url=config.base_url or None,
        timeout=config.timeout_seconds,
    )

    marker = f"BRAIN_TUMOR_PROVIDER_PROBE_{uuid4().hex[:12]}"
    temp_file_path: Path | None = None
    file_id: str | None = None
    vector_store_id: str | None = None

    try:
        response = client.responses.create(
            model=config.model,
            input="Reply with OK only.",
            max_output_tokens=20,
        )
        output_text = _extract_output_text(response)
        report.steps.append(
            ProbeStepResult(
                name="responses_basic",
                ok=True,
                detail="Responses API 调用成功。",
                payload={"output_text": output_text},
            )
        )
    except Exception as exc:
        report.steps.append(
            ProbeStepResult(
                name="responses_basic",
                ok=False,
                detail=f"Responses API 调用失败: {exc}",
            )
        )
        return _finalize_report(report, config)

    try:
        temp_file_path = _write_probe_file(marker)
        with temp_file_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="assistants")
        file_id = getattr(uploaded, "id", None)
        report.file_id = file_id
        report.steps.append(
            ProbeStepResult(
                name="files_create",
                ok=bool(file_id),
                detail="Files API 上传成功。" if file_id else "Files API 返回中缺少 file_id。",
                payload={"file_id": file_id, "status": getattr(uploaded, "status", None)},
            )
        )
        if not file_id:
            return _finalize_report(report, config)
    except Exception as exc:
        report.steps.append(
            ProbeStepResult(
                name="files_create",
                ok=False,
                detail=f"Files API 上传失败: {exc}",
            )
        )
        return _finalize_report(report, config, temp_file_path=temp_file_path)

    try:
        vector_store = client.vector_stores.create(name=f"brain-tumor-provider-probe-{uuid4().hex[:8]}")
        vector_store_id = getattr(vector_store, "id", None)
        report.vector_store_id = vector_store_id
        report.steps.append(
            ProbeStepResult(
                name="vector_store_create",
                ok=bool(vector_store_id),
                detail="vector_store 创建成功。" if vector_store_id else "vector_store 创建返回中缺少 id。",
                payload={"vector_store_id": vector_store_id},
            )
        )
        if not vector_store_id:
            return _finalize_report(report, config, temp_file_path=temp_file_path, client=client, file_id=file_id)
    except Exception as exc:
        report.steps.append(
            ProbeStepResult(
                name="vector_store_create",
                ok=False,
                detail=f"vector_store 创建失败: {exc}",
            )
        )
        return _finalize_report(report, config, temp_file_path=temp_file_path, client=client, file_id=file_id)

    try:
        vector_store_file = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
        initial_status = getattr(vector_store_file, "status", None)
        polled_status = initial_status
        start = time.time()
        while polled_status not in TERMINAL_VECTOR_STORE_STATUSES:
            if time.time() - start > config.poll_timeout_seconds:
                raise TimeoutError("等待 vector_store 文件处理超时。")
            time.sleep(config.poll_interval_seconds)
            polled = client.vector_stores.files.retrieve(
                vector_store_id=vector_store_id,
                file_id=file_id,
            )
            polled_status = getattr(polled, "status", None)

        ok = polled_status == "completed"
        report.steps.append(
            ProbeStepResult(
                name="vector_store_attach_and_poll",
                ok=ok,
                detail=f"vector_store 文件状态: {polled_status}",
                payload={"initial_status": initial_status, "final_status": polled_status},
            )
        )
        if not ok:
            return _finalize_report(
                report,
                config,
                temp_file_path=temp_file_path,
                client=client,
                file_id=file_id,
                vector_store_id=vector_store_id,
            )
    except Exception as exc:
        report.steps.append(
            ProbeStepResult(
                name="vector_store_attach_and_poll",
                ok=False,
                detail=f"vector_store 挂载或轮询失败: {exc}",
            )
        )
        return _finalize_report(
            report,
            config,
            temp_file_path=temp_file_path,
            client=client,
            file_id=file_id,
            vector_store_id=vector_store_id,
        )

    try:
        rag_response = client.responses.create(
            model=config.model,
            input=f"Return only the probe token from the document. If not found, return NOT_FOUND.",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": 3,
                }
            ],
            include=["file_search_call.results"],
            max_output_tokens=80,
        )
        output_text = _extract_output_text(rag_response)
        results = _extract_file_search_results(rag_response)
        result_text = json.dumps(results, ensure_ascii=False)
        ok = marker in output_text or marker in result_text or bool(results)
        report.steps.append(
            ProbeStepResult(
                name="responses_file_search",
                ok=ok,
                detail="file_search 端到端调用成功。" if ok else "file_search 调用返回成功，但没有检测到检索命中。",
                payload={
                    "output_text": output_text,
                    "result_count": len(results),
                    "results": results,
                },
            )
        )
    except Exception as exc:
        report.steps.append(
            ProbeStepResult(
                name="responses_file_search",
                ok=False,
                detail=f"file_search 端到端调用失败: {exc}",
            )
        )

    return _finalize_report(
        report,
        config,
        temp_file_path=temp_file_path,
        client=client,
        file_id=file_id,
        vector_store_id=vector_store_id,
    )


def _finalize_report(
    report: ProviderProbeReport,
    config: ProviderProbeConfig,
    *,
    temp_file_path: Path | None = None,
    client: OpenAI | None = None,
    file_id: str | None = None,
    vector_store_id: str | None = None,
) -> ProviderProbeReport:
    cleanup_payload: dict[str, Any] = {}
    if temp_file_path and temp_file_path.exists():
        temp_file_path.unlink(missing_ok=True)

    if client and not config.keep_artifacts:
        cleanup_payload.update(_cleanup_probe_artifacts(client, file_id=file_id, vector_store_id=vector_store_id))

    if cleanup_payload:
        report.steps.append(
            ProbeStepResult(
                name="cleanup",
                ok=cleanup_payload.get("cleanup_ok", False),
                detail=cleanup_payload.get("detail", "cleanup finished"),
                payload=cleanup_payload,
            )
        )

    report.overall_ok = all(step.ok for step in report.steps if step.name != "cleanup")
    report.report_path = str(_write_report_file(report, config.report_dir))
    return report


def _cleanup_probe_artifacts(client: OpenAI, *, file_id: str | None, vector_store_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"cleanup_ok": True}
    messages: list[str] = []

    if vector_store_id and hasattr(client.vector_stores, "delete"):
        try:
            client.vector_stores.delete(vector_store_id)
            messages.append("vector_store deleted")
        except Exception as exc:
            payload["cleanup_ok"] = False
            messages.append(f"vector_store delete failed: {exc}")

    if file_id and hasattr(client.files, "delete"):
        try:
            client.files.delete(file_id)
            messages.append("file deleted")
        except Exception as exc:
            payload["cleanup_ok"] = False
            messages.append(f"file delete failed: {exc}")

    payload["detail"] = "; ".join(messages) if messages else "no cleanup endpoints available"
    payload["file_id"] = file_id
    payload["vector_store_id"] = vector_store_id
    return payload


def _write_probe_file(marker: str) -> Path:
    with NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write("Brain tumor MRI provider probe file.\n")
        handle.write(f"Probe token: {marker}\n")
        return Path(handle.name)


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                chunks.append(getattr(content, "text", ""))
    return "".join(chunks).strip()


def _extract_file_search_results(response: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "file_search_call":
            continue
        for result in getattr(item, "results", []) or []:
            if hasattr(result, "model_dump"):
                rows.append(result.model_dump())
            elif hasattr(result, "__dict__"):
                rows.append({k: v for k, v in vars(result).items() if not k.startswith("_")})
            elif isinstance(result, dict):
                rows.append(result)
            else:
                rows.append({"value": str(result)})
    return rows


def _write_report_file(report: ProviderProbeReport, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"provider_probe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(report.to_json(), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.env_file:
        load_env_file(args.env_file)

    config = build_config_from_args(args)
    report = run_provider_probe(config)

    print(f"Provider base URL: {report.base_url or 'official-default'}")
    print(f"Model: {report.model}")
    print(f"Overall OK: {report.overall_ok}")
    for step in report.steps:
        status = "PASS" if step.ok else "FAIL"
        print(f"[{status}] {step.name}: {step.detail}")
    print(f"Report: {report.report_path}")
    return 0 if report.overall_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())