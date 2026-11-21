
"""Point d'entrée Streamlit Cloud pour RAG FOMC."""

from pathlib import Path
import runpy

from huggingface_hub import snapshot_download
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
E5_DIR = (
    ROOT / ".cache" / "huggingface" / "hub"
    / "models--intfloat--e5-small-v2"
    / "snapshots" / E5_REVISION
)

if not (E5_DIR / "model.safetensors").exists():
    with st.spinner("Chargement du modèle E5 préentraîné..."):
        snapshot_download(
            repo_id="intfloat/e5-small-v2",
            revision=E5_REVISION,
            cache_dir=str(ROOT / ".cache" / "huggingface" / "hub"),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.safetensors",
                "1_Pooling/*",
            ],
        )

runpy.run_path(
    str(ROOT / "app" / "streamlit_app.py"),
    run_name="__main__",
)
