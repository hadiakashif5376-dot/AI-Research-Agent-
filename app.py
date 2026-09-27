import time
import re
import traceback
import streamlit as st
from litellm.exceptions import RateLimitError
from agent import build_research_crew

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("Powered by CrewAI + Groq (Llama 4 Scout) + DuckDuckGo search")

# --- API key comes only from Streamlit Cloud secrets (Settings -> Secrets) ---
groq_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_key:
    st.error(
        "GROQ_API_KEY not found in Streamlit Secrets. "
        "Go to your app's Settings -> Secrets and add:\n\n"
        'GROQ_API_KEY = "your_key_here"'
    )

topic = st.text_input("What do you want me to research?", placeholder="e.g. Latest trends in solid-state batteries")


def run_with_retry(topic: str, groq_key: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            crew = build_research_crew(topic, groq_key)
            return crew.kickoff()
        except RateLimitError as e:
            match = re.search(r"try again in ([\d.]+)s", str(e))
            wait_seconds = float(match.group(1)) + 5 if match else 20
            if attempt < max_retries - 1:
                st.info(f"Groq rate limit hit — waiting {wait_seconds:.0f}s and retrying "
                         f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                raise
    return None


if st.button("Research", type="primary", disabled=not groq_key):
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
