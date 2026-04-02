# 本文件提供 Streamlit 演示界面，用于查看知识库状态、发起问答和展示证据片段。

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st



DEFAULT_API_URL = os.getenv('BMAGENT_BACKEND_URL', 'http://127.0.0.1:8000')


# 执行 GET 请求并返回 JSON，失败时返回 None 供 UI 容错。
def get_json(base_url: str, path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f'{base_url}{path}', timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return response.json()


# 执行 POST 请求并返回 JSON，出错时由 UI 层捕获异常。
def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f'{base_url}{path}', json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title='Brain Tumor MRI Assistant', layout='wide')
st.title('Brain Tumor MRI 智能助手')
st.caption('Local RAG + Yunwu/OpenAI-compatible Responses API')

if 'backend_url' not in st.session_state:
    st.session_state.backend_url = DEFAULT_API_URL
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header('连接设置')
    st.session_state.backend_url = st.text_input('Backend URL', value=st.session_state.backend_url)
    health = get_json(st.session_state.backend_url, '/api/healthz')
    if health:
        st.success(f"后端已连接，模型: {health['model']}")
        st.caption(f"检索模式: {health['retrieval_mode']}")
    else:
        st.error('后端不可达，请先启动 FastAPI。')

    st.header('本地知识库')
    kb_status = get_json(st.session_state.backend_url, '/api/kb/status')
    if kb_status:
        st.write(
            {
                'knowledge_base_id': kb_status['knowledge_base_id'],
                'knowledge_base_name': kb_status['knowledge_base_name'],
                'total_documents': kb_status['total_documents'],
                'indexed_documents': kb_status['indexed_documents'],
                'chunk_count': kb_status['chunk_count'],
                'failed_documents': kb_status['failed_documents'],
            }
        )

    source_dir = st.text_input('文档目录', value='data/knowledge_base/source')
    knowledge_base_name = st.text_input('知识库名称', value='brain-tumor-mri-kb-local')
    chunk_size_chars = st.number_input('chunk size (chars)', min_value=400, value=1400, step=100)
    chunk_overlap_chars = st.number_input('chunk overlap (chars)', min_value=0, value=250, step=50)
    dry_run = st.toggle('仅 dry run', value=False)

    if st.button('构建本地索引', use_container_width=True):
        try:
            result = post_json(
                st.session_state.backend_url,
                '/api/kb/sync',
                {
                    'source_dir': source_dir,
                    'knowledge_base_name': knowledge_base_name,
                    'chunk_size_chars': int(chunk_size_chars),
                    'chunk_overlap_chars': int(chunk_overlap_chars),
                    'dry_run': dry_run,
                },
            )
            st.success(
                f"索引完成: kb={result['knowledge_base_id']}, chunks={result['chunk_count']}, indexed={result['indexed_files']}, failed={result['failed_files']}"
            )
            kb_status = get_json(st.session_state.backend_url, '/api/kb/status')
        except requests.HTTPError as exc:
            st.error(exc.response.text)
        except requests.RequestException as exc:
            st.error(str(exc))

    st.header('会话')
    current_session_label = st.session_state.session_id or '尚未创建'
    st.text_input('当前 session_id', value=current_session_label, disabled=True)
    if st.button('新建会话', use_container_width=True):
        try:
            session = post_json(st.session_state.backend_url, '/api/sessions', {})
            st.session_state.session_id = session['session_id']
            st.session_state.messages = []
            st.success(f"已创建会话: {session['session_id']}")
        except requests.HTTPError as exc:
            st.error(exc.response.text)
        except requests.RequestException as exc:
            st.error(str(exc))

for item in st.session_state.messages:
    with st.chat_message('user'):
        st.write(item['question'])
    with st.chat_message('assistant'):
        answer = item['answer']
        st.markdown(f"**摘要**\n\n{answer['answer_summary']}")
        st.write(answer['answer_detail'])

        if item.get('retrieval_queries'):
            st.markdown('**检索查询**')
            for query in item['retrieval_queries']:
                st.write(f'- {query}')

        if answer['key_points']:
            st.markdown('**关键点**')
            for point in answer['key_points']:
                st.write(f'- {point}')

        if answer['imaging_features']:
            st.markdown('**影像特征**')
            for feature in answer['imaging_features']:
                st.json(feature)

        if answer['differential_diagnosis']:
            st.markdown('**影像鉴别**')
            for diag in answer['differential_diagnosis']:
                st.json(diag)

        if answer['sequence_meaning']:
            st.markdown('**序列意义**')
            for seq in answer['sequence_meaning']:
                st.json(seq)

        if answer['evidence']:
            st.markdown('**证据引用**')
            for evidence in answer['evidence']:
                st.json(evidence)

        if item['retrieved_snippets']:
            st.markdown('**本地检索片段**')
            for idx, snippet in enumerate(item['retrieved_snippets'], start=1):
                label = f"[{idx}] {snippet['file_name']}"
                if snippet.get('page_hint'):
                    label += f" | {snippet['page_hint']}"
                with st.expander(label, expanded=idx == 1):
                    if snippet.get('score') is not None:
                        st.caption(f"score={snippet['score']}")
                    st.write(snippet['snippet'])

        if answer['limitations']:
            st.markdown('**局限性**')
            for limitation in answer['limitations']:
                st.write(f'- {limitation}')

        if answer['follow_up_questions']:
            st.markdown('**建议追问**')
            for follow_up in answer['follow_up_questions']:
                st.write(f'- {follow_up}')

        st.caption(answer['safety_note'])

question = st.chat_input('例如：胶质母细胞瘤在 MRI 上的典型表现是什么？')
if question:
    with st.chat_message('user'):
        st.write(question)

    payload = {
        'session_id': st.session_state.session_id,
        'question': question,
        'use_query_rewrite': True,
    }
    if kb_status and kb_status.get('knowledge_base_id'):
        payload['knowledge_base_id'] = kb_status['knowledge_base_id']

    try:
        response = post_json(st.session_state.backend_url, '/api/qa', payload)
        st.session_state.session_id = response['session']['session_id']
        st.session_state.messages.append(
            {
                'question': question,
                'answer': response['answer'],
                'retrieval_queries': response.get('retrieval_queries', []),
                'retrieved_snippets': response.get('retrieved_snippets', []),
            }
        )
        st.rerun()
    except requests.HTTPError as exc:
        with st.chat_message('assistant'):
            st.error(exc.response.text)
    except requests.RequestException as exc:
        with st.chat_message('assistant'):
            st.error(str(exc))
