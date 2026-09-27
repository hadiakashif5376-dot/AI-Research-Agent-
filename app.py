import re
import time
import traceback
import streamlit as st
from litellm.exceptions import RateLimitError, BadRequestError
from agent import build_research_crew

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")

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

TOPIC_OPTIONS = [
    "Type your own topic...",
    "The impact of AI on software jobs in 2026",
    "Renewable energy trends in 2026",
    "The future of remote work",
    "Electric vehicle adoption in Pakistan",
]

# --- Session state defaults ---
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {topic, report, sources, length}
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None


def split_report_and_sources(text: str):
    """Splits the model's markdown output into (report_body, sources) at the
    '## Sources' (or similar) heading, if present."""
    match = re.search(r"^#{1,3}\s*Sources\b.*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        report = text[: match.start()].strip()
        sources = text[match.end():].strip()
        return report, sources
    return text, None


def run_with_retry(topic: str, groq_key: str, report_length: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            crew = build_research_crew(topic, groq_key, report_length)
            return crew.kickoff()
        except RateLimitError as e:
            match = re.search(r"try again in ([\d.]+)s", str(e))
            wait_seconds = float(match.group(1)) + 8 if match else 25
            if attempt < max_retries - 1:
                st.info(f"Rate limit hit — waiting {wait_seconds:.0f}s and retrying "
                         f"({attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                raise
                        except BadRequestError as e:
            # The model occasionally misbehaves around tool calls — either
            # hallucinating a tool that doesn't exist, or trying to call a
            # tool right when it's being forced to give a final answer.
            # Both are non-deterministic — simply retrying almost always works.
            msg = str(e)
            tool_issue = "tool call validation failed" in msg or "Tool choice is none" in msg
            if tool_issue and attempt < max_retries - 1:
                st.info(f"Agent tried an invalid action — retrying ({attempt + 1}/{max_retries})...")
                continue
            else:
                raise
    return None


# --- Sidebar ---
with st.sidebar:
    st.header("About")
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

    st.markdown("---")
    st.subheader("History")
    if not st.session_state.history:
        st.caption("No past searches yet.")
    else:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
        for entry in reversed(st.session_state.history):
            with st.expander(entry["topic"]):
                st.markdown(entry["report"])
                if entry["sources"]:
                    st.markdown("**Sources**")
                    st.markdown(entry["sources"])

# --- Main content ---
st.title("AI Research Agent")
st.caption("Single-agent researcher built with CrewAI · Groq (openai/gpt-oss-120b) · DuckDuckGo Search")

if not groq_key:
    st.warning("Add GROQ_API_KEY in Streamlit Secrets to use this app.")

# --- Combined topic field: pick an example OR type your own ---
topic_choice = st.selectbox("What topic should the agent research?", TOPIC_OPTIONS, key="topic_choice")

if topic_choice == TOPIC_OPTIONS[0]:
    topic = st.text_input(
        "Your topic",
        key="topic_input",
        placeholder="e.g. The impact of AI on software jobs in 2026",
        label_visibility="collapsed",
    )
else:
    topic = topic_choice

length_choice = st.radio("Report length", ["Quick summary", "Detailed report"], index=1, horizontal=True)
report_length = "quick" if length_choice == "Quick summary" else "detailed"

if st.button("Run Research", type="primary", disabled=not groq_key):
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        with st.spinner("Agent is researching... this can take 20-60 seconds."):
            try:
                result = run_with_retry(topic, groq_key, report_length)
                report_text, sources_text = split_report_and_sources(str(result))

                st.session_state.last_topic = topic
                st.session_state.last_report = report_text
                st.session_state.last_sources = sources_text
                st.session_state.history.append(
                    {"topic": topic, "report": report_text, "sources": sources_text, "length": report_length}
                )
            except Exception:
                st.error("Something went wrong. Full details below:")
                st.code(traceback.format_exc())

# --- Show the latest result (persists across reruns, e.g. opening the sidebar) ---
if st.session_state.last_report:
    st.success("Done!")
    st.markdown(st.session_state.last_report)

    if st.session_state.last_sources:
        with st.expander("Sources"):
            st.markdown(st.session_state.last_sources)

    st.download_button(
        label="Download report as Markdown",
        data=st.session_state.last_report
        + ("\n\n## Sources\n" + st.session_state.last_sources if st.session_state.last_sources else ""),
        file_name="research_report.md",
        mime="text/markdown",
    )
