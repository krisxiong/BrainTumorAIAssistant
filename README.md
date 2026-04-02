# Brain Tumor MRI Assistant

这是一个面向脑肿瘤 MRI 场景的证据增强问答助手。当前版本已经从 OpenAI 托管 `file_search` 架构切换为“本地 RAG + Yunwu/OpenAI-compatible Responses API 生成”，也就是：

- 文档解析、切分、建索引、检索都在本地完成
- Yunwu 只负责 query rewrite 和最终答案生成
- 前端展示结构化回答、证据引用和检索片段

## 当前架构

```text
教材 / 综述 / 论文 PDF / DOCX / Markdown
                |
                v
      本地文档解析与切分
                |
                v
        本地 BM25 索引检索
                |
                v
 FastAPI 组装 prompt + 证据 + 会话上下文
                |
                v
   Yunwu / OpenAI-compatible Responses API
                |
                v
      结构化答案 + 证据引用 + 检索片段
                |
                v
             Streamlit
```

## 为什么改成本地 RAG

Yunwu 可以兼容基础 `Responses API` 和 function calling，但你这版项目真正依赖的托管能力是：

- `Files API`
- `vector_store`
- `file_search`

探针测试已经证明这些平台能力在 Yunwu 上不稳定或不可用，所以当前版本改为：

- 本地自己做解析、切分、索引、检索
- 兼容接口只做生成

这样既能继续用 Yunwu，也更适合你自己深入理解 RAG 的实际实现方式。

## 核心模块

- `src/bmagent_rag/local_rag.py`
  - 文档读取、文本清洗、chunk 切分、本地 BM25 检索
- `src/bmagent_rag/sync.py`
  - 扫描本地知识库、计算 SHA256、重建本地索引、写 manifest
- `src/bmagent_rag/qa_service.py`
  - query rewrite、检索、拼接证据、调用 Yunwu 生成结构化答案
- `src/bmagent_rag/qa_api.py`
  - FastAPI 接口
- `frontend/streamlit_app.py`
  - 演示前端
- `src/bmagent_rag/literature.py`
  - 安全版文献候选采集与 OA 下载
- `src/bmagent_rag/provider_probe.py`
  - OpenAI-compatible 平台能力探针

## 快速开始

建议使用 Python 3.11：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
py -3.11 -m pip install -e .[dev]
Copy-Item .env.example .env
```

`.env` 最小配置示例：

```env
OPENAI_API_KEY=你的 Yunwu 新密钥
OPENAI_BASE_URL=https://yunwu.ai/v1
OPENAI_MODEL=gpt-4.1
OPENAI_REASONING_EFFORT=medium
OPENAI_TEXT_VERBOSITY=low
OPENAI_MAX_OUTPUT_TOKENS=1400

BMAGENT_KB_SOURCE_DIR=data/knowledge_base/source
BMAGENT_KB_STATE_DIR=data/knowledge_base/state
BMAGENT_KB_NAME=brain-tumor-mri-kb-local
BMAGENT_KB_CHUNK_SIZE_CHARS=1400
BMAGENT_KB_CHUNK_OVERLAP_CHARS=250

BMAGENT_BACKEND_URL=http://127.0.0.1:8000
BMAGENT_CONTACT_EMAIL=
OPENALEX_MAILTO=
NCBI_API_KEY=
```

注意：

- 不要继续使用之前泄露过的旧密钥，应当先在 Yunwu 后台撤销并重建。
- `OPENAI_BASE_URL` 指向 Yunwu 的 OpenAI-compatible 地址，例如 `https://yunwu.ai/v1`。
- 当前版本不再依赖 `OPENAI_VECTOR_STORE_ID`。

## 本地知识库准备

把资料放到 `data/knowledge_base/source`。推荐先放：

- 1 本脑肿瘤 MRI 教材
- 3 到 5 篇高质量综述
- 5 到 10 篇核心论文

支持的常见格式：

- `.pdf`
- `.docx`
- `.txt`
- `.md`
- `.html`
- `.csv`
- `.json`

## 构建本地索引

```powershell
py -3.11 scripts\sync_knowledge_base.py --dry-run
py -3.11 scripts\sync_knowledge_base.py
```

这一步会完成：

- 扫描文档目录
- 计算文件哈希
- 提取文本
- 按字符窗口切 chunk
- 建立本地 BM25 索引
- 写入 `manifest.json` 和 `local_index.json`

## 启动后端与前端

启动 FastAPI：

```powershell
uvicorn app.main:app --reload
```

启动 Streamlit：

```powershell
streamlit run frontend/streamlit_app.py
```

## 主要接口

- `GET /api/healthz`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/kb/status`
- `POST /api/kb/sync`
- `POST /api/qa`

## 安全版文献采集

先拉候选元数据，不直接抓付费全文：

```powershell
py -3.11 scripts\search_literature_candidates.py "glioblastoma MRI review" --max-results 20 --from-year 2020 --reviews-only
```

如果候选结果合适，再只下载开放获取 PDF：

```powershell
py -3.11 scripts\search_literature_candidates.py "glioblastoma MRI review" --max-results 20 --from-year 2020 --reviews-only --download-open-access
```

## 平台兼容性探针

如果你想继续测试某个 OpenAI-compatible 服务到底支持到哪一层：

```powershell
py -3.11 scripts\probe_provider_compat.py --base-url "https://yunwu.ai/v1" --model "gpt-4.1"
```

这个探针会检查：

- 基础 `Responses API`
- `Files API`
- `vector_store`
- `file_search`

当前经验结论是：Yunwu 更适合做生成，不适合承担托管 RAG 平台能力。

## 你现在可以重点研究的 RAG 环节

如果你的目标是顺着项目把 RAG 学透，建议按下面顺序读代码：

1. `src/bmagent_rag/local_rag.py`
   - 看文本如何被切成 chunk
   - 看 BM25 如何给片段打分
2. `src/bmagent_rag/sync.py`
   - 看知识库如何扫描、去重、重建本地索引
3. `src/bmagent_rag/qa_service.py`
   - 看 query rewrite、检索、prompt 组装、结构化回答是怎么串起来的
4. `frontend/streamlit_app.py`
   - 看答案、证据、检索片段如何展示给用户

## 已验证状态

本地回归已通过：

```powershell
py -3.11 -m pytest
py -3.11 -m compileall app frontend scripts src tests
```

当前测试结果：`16 passed`