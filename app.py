import streamlit as st
from agent import build_research_crew

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("Powered by CrewAI + Groq (openai/gpt-oss-120b) + DuckDuckGo search")

# --- API key comes only from Streamlit Cloud secrets (Settings -> Secrets) ---
groq_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_key:
    st.error(
        "GROQ_API_KEY not found in Streamlit Secrets. "
        "Go to your app's Settings -> Secrets and add:\n\n"
        'GROQ_API_KEY = "your_key_here"'
    )

topic = st.text_input("What do you want me to research?", placeholder="e.g. Latest trends in solid-state batteries")

if st.button("Research", type="primary", disabled=not groq_key):
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        with st.spinner("Agent is researching... this can take 20-60 seconds."):
            try:
                crew = build_research_crew(topic, groq_key)
                result = crew.kickoff()
                st.success("Done!")
                st.markdown(str(result))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
