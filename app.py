import time
import re
import traceback
import streamlit as st
from litellm.exceptions import RateLimitError
from agent import build_research_crew

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")

# --- Light custom styling ---
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        color: white;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

groq_key = st.secrets.get("GROQ_API_KEY", "")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ About")
    if groq_key:
        st.success("Groq API key loaded from secrets")
    else:
        st.error("Groq API key missing")

    st.markdown("---")
    st.markdown("**Model:** `openai/gpt-oss-120b` (via Groq)")
    st.markdown("**Search:** DuckDuckGo (free, no key needed)")
    st.markdown("---")
    st.caption(
        "This app sends your topic to a Groq-hosted LLM and does live "
        "DuckDuckGo web searches. Don't enter sensitive information."
    )

# --- Main content ---
st.title("🔎 AI Research Agent")
st.caption("Single-agent researcher built with CrewAI · Groq (openai/gpt-oss-120b) · DuckDuckGo Search")

if not groq_key:
    st.warning("Add GROQ_API_KEY in Streamlit Secrets to use this app.")

topic = st.text_input("What topic should the agent research?", placeholder="e.g. The impact of AI on software jobs in 2026")


def run_with_retry(topic: str, groq_key: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            crew = build_research_crew(topic, groq_key)
            return crew.kickoff()
        except RateLimitError as e:
            match = re.search(r"try again in ([\d.]+)s", str(e))
            wait_seconds = float(match.group(1)) + 8 if match else 25
            if attempt < max_retries - 1:
                st.info(f"Groq rate limit hit — waiting {wait_seconds:.0f}s and retrying "
                         f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                raise
    return None


if st.button("Run Research", type="primary", disabled=not groq_key):
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        with st.spinner("Agent is researching... this can take 20-60 seconds."):
            try:
                result = run_with_retry(topic, groq_key)
                st.success("Done!")
                st.markdown(str(result))
            except Exception:
                st.error("Something went wrong. Full details below:")
                st.code(traceback.format_exc())
