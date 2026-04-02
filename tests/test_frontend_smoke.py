from pathlib import Path


def test_streamlit_frontend_contains_key_controls() -> None:
    path = Path('frontend/streamlit_app.py')
    content = path.read_text(encoding='utf-8')

    assert 'st.chat_input' in content
    assert '/api/qa' in content
    assert '/api/kb/status' in content
    assert 'Local RAG' in content
    assert 'knowledge_base_id' in content