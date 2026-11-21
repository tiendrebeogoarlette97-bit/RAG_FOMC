"""Cloud entry point: keep the original src algorithms."""
from pathlib import Path
from functools import partial
import os
import runpy
import sys
from urllib.parse import urlsplit
import streamlit as st
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import CONFIG, model_dir


def setting(name, default=''):
    try:
        return str(st.secrets.get(name, os.environ.get(name, default))).strip()
    except FileNotFoundError:
        return os.environ.get(name, default).strip()


@st.cache_resource(show_spinner=False)
def prepare_models():
    required = ['data/processed/chunks.jsonl', 'indexes/bm25/tokens.jsonl',
                'indexes/bm25/metadata.json', 'indexes/dense/vectors.npy',
                'indexes/dense/chunk_ids.json', 'indexes/dense/metadata.json']
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError('Fichiers absents du depot deploye : ' + ', '.join(missing))
    for kind in ('dense', 'colbert', 'cross_encoder'):
        snapshot_download(repo_id=CONFIG[kind + '_model'], revision=CONFIG[kind + '_revision'],
            local_dir=str(model_dir(kind)), allow_patterns=[
                'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'vocab.txt', 'model.safetensors',
                'onnx_config.json', 'modules.json', 'sentence_bert_config.json',
                '1_Pooling/config.json'])
    from build_indexes import load_indexes
    load_indexes()
    return True


def main():
    # The original UI calls set_page_config: do not call it a second time here.
    try:
        prepare_models()
    except Exception:
        import logging
        logging.exception('Cloud model/index initialization failed')
        st.error('Initialisation impossible. Consulte les logs Streamlit : modeles, index ou memoire du serveur.')
        st.stop()
    endpoint = setting('OLLAMA_ENDPOINT')
    model = setting('OLLAMA_MODEL', 'qwen2.5:3b')
    if endpoint:
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.hostname in ('localhost', '127.0.0.1', '::1'):
            st.error('OLLAMA_ENDPOINT doit etre une adresse HTTPS distante, sans localhost.')
            st.stop()
    import generate
    import coverage_guard
    for module, name in ((generate, 'generate'), (coverage_guard, 'generate_with_coverage')):
        original_name = '_cloud_original_' + name
        if not hasattr(module, original_name):
            setattr(module, original_name, getattr(module, name))
        original = getattr(module, original_name)
        setattr(module, name, partial(original, endpoint=endpoint) if endpoint else original)
    groq_available = bool(setting('GROQ_API_KEY'))
    if endpoint:
        st.session_state.setdefault('model_name', model)
    # Never overwrite the user's use_llm toggle on each Streamlit rerun.
    # Groq provides cloud generation even without an Ollama endpoint.
    runpy.run_path(str(ROOT / 'app/streamlit_app.py'), run_name='__main__')
    if not endpoint and not groq_available:
        st.warning('Pour activer Qwen sur Streamlit Cloud, configure GROQ_API_KEY dans les Secrets.')


if __name__ == '__main__':
    main()
