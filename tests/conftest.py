import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import get_type_hints

import pydantic


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if not hasattr(pydantic, "ConfigDict"):
    pydantic.ConfigDict = dict  # type: ignore[attr-defined]

BaseModel = pydantic.BaseModel

if not hasattr(BaseModel, "model_validate"):
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        return cls.parse_obj(obj)

    BaseModel.model_validate = model_validate  # type: ignore[assignment]

if not hasattr(BaseModel, "model_validate_json"):
    @classmethod
    def model_validate_json(cls, json_data, *args, **kwargs):
        return cls.parse_raw(json_data)

    BaseModel.model_validate_json = model_validate_json  # type: ignore[assignment]

if not hasattr(BaseModel, "model_dump"):
    def model_dump(self, *args, **kwargs):
        return self.dict(*args, **kwargs)

    BaseModel.model_dump = model_dump  # type: ignore[assignment]

if not hasattr(BaseModel, "model_dump_json"):
    def model_dump_json(self, *args, **kwargs):
        return self.json(*args, **kwargs)

    BaseModel.model_dump_json = model_dump_json  # type: ignore[assignment]

if not hasattr(BaseModel, "model_json_schema"):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        return cls.schema(*args, **kwargs)

    BaseModel.model_json_schema = model_json_schema  # type: ignore[assignment]

if "fastapi" not in sys.modules:
    fake_fastapi = ModuleType("fastapi")
    fake_fastapi_testclient = ModuleType("fastapi.testclient")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.routes = []
            self.title = kwargs.get("title")
            self.version = kwargs.get("version")

        def get(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append(SimpleNamespace(path=path, endpoint=func, method="GET"))
                return func

            return decorator

        def post(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append(SimpleNamespace(path=path, endpoint=func, method="POST"))
                return func

            return decorator

    class _FakeResponse:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self):
            return self._payload

    class TestClient:
        __test__ = False

        def __init__(self, app):
            self.app = app

        def get(self, path: str):
            return self._request("GET", path, None)

        def post(self, path: str, json=None):
            return self._request("POST", path, json or {})

        def _request(self, method: str, path: str, json_payload):
            for route in self.app.routes:
                params = _match_path(route.path, path)
                if route.method == method and params is not None:
                    try:
                        kwargs = _build_kwargs(route.endpoint, params, json_payload)
                        result = route.endpoint(**kwargs)
                        if hasattr(result, "model_dump"):
                            return _FakeResponse(200, result.model_dump())
                        if isinstance(result, dict):
                            return _FakeResponse(200, result)
                        return _FakeResponse(200, result)
                    except HTTPException as exc:
                        return _FakeResponse(exc.status_code, {"detail": exc.detail})
            return _FakeResponse(404, {"detail": "not found"})

    def _match_path(route_path: str, actual_path: str):
        pattern = re.sub(r"\{([^{}]+)\}", r"(?P<\1>[^/]+)", route_path)
        match = re.fullmatch(pattern, actual_path)
        if not match:
            return None
        return match.groupdict()

    def _build_kwargs(endpoint, path_params, json_payload):
        kwargs = {}
        signature = inspect.signature(endpoint)
        try:
            resolved_hints = get_type_hints(endpoint)
        except Exception:
            resolved_hints = {}

        for name, parameter in signature.parameters.items():
            annotation = resolved_hints.get(name, parameter.annotation)
            if name in path_params:
                kwargs[name] = path_params[name]
                continue
            if annotation is inspect._empty:
                kwargs[name] = json_payload
                continue
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                kwargs[name] = annotation.model_validate(json_payload)
                continue
            kwargs[name] = json_payload
        return kwargs

    fake_fastapi.FastAPI = FastAPI
    fake_fastapi.HTTPException = HTTPException
    fake_fastapi_testclient.TestClient = TestClient
    sys.modules["fastapi"] = fake_fastapi
    sys.modules["fastapi.testclient"] = fake_fastapi_testclient

if "openai" not in sys.modules:
    fake_openai = ModuleType("openai")

    class _StubVectorStoreFiles:
        def create(self, **kwargs):
            return SimpleNamespace(id=kwargs.get("file_id", "vector_store_file_stub"), status="completed")

        def retrieve(self, **kwargs):
            return SimpleNamespace(id=kwargs.get("file_id", "vector_store_file_stub"), status="completed")

    class _StubVectorStores:
        def __init__(self):
            self.files = _StubVectorStoreFiles()

        def create(self, **kwargs):
            return SimpleNamespace(id="vector_store_stub", **kwargs)

    class _StubFiles:
        def create(self, **kwargs):
            return SimpleNamespace(id="file_stub", status="uploaded")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            self.files = _StubFiles()
            self.vector_stores = _StubVectorStores()
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(id="response_stub", output=[]))

    fake_openai.OpenAI = OpenAI
    sys.modules["openai"] = fake_openai