import streamlit as st
import sys
import os
import requests


def check_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def render_chat(project_root: str):
    st.markdown("---")
    st.markdown("##### Ask about your model")

    if not check_ollama_running():
        st.warning(
            "Ollama is not running. "
            "Start it with `ollama serve` in a terminal to enable chat."
        )
        return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    col1, col2, col3 = st.columns(3)
    trigger_question = None

    if col1.button("Best model?",       use_container_width=True):
        trigger_question = "Which model performed best and why?"
    if col2.button("Retrain now?",      use_container_width=True):
        trigger_question = "Based on the latest drift report, should I retrain?"
    if col3.button("Features drifted?", use_container_width=True):
        trigger_question = "Which features drifted the most?"

    if trigger_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": trigger_question}
        )
        sys.path.insert(0, project_root)
        from src.rag.retriever import ask
        answer = ask(trigger_question)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about drift, models, features..."):
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                sys.path.insert(0, project_root)
                from src.rag.retriever import ask
                answer = ask(prompt)
            st.write(answer)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )
