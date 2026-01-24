import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NLP AI Copilot", layout="wide")

st.title("NLP AI Copilot")
st.success("If you see this, Streamlit is running correctly 🎉")

kb_dir = Path("data/knowledge_base")
ds_dir = Path("data/datasets")

st.subheader("Project folders")

col1, col2 = st.columns(2)

with col1:
    st.write("📄 Knowledge Base")
    st.code(str(kb_dir))
    st.write("Files:", [p.name for p in kb_dir.glob("*")])

with col2:
    st.write("📊 Datasets")
    st.code(str(ds_dir))
    st.write("Files:", [p.name for p in ds_dir.glob("*")])
