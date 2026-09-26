# AI Research Agent

A single-agent CrewAI app that researches any topic using free DuckDuckGo
search and Groq's `openai/gpt-oss-120b` model, with a Streamlit UI.

This version is set up to run only on Streamlit Community Cloud, using
Streamlit Secrets for the Groq API key.

## Project structure

```
ai-research-agent/
├── app.py                  # Streamlit UI — entry point
├── agent.py                # Agent + task + crew + the DuckDuckGo search tool
├── requirements.txt
└── README.md
```

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repo (use `git push`, not the drag-and-drop
   uploader on github.com — it can silently skip files/folders).
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app,"
   and point it at your repo with main file `app.py`.
3. In **Advanced settings** during deploy (or **Settings** afterward), set
   **Python version** to **3.11** or **3.12**. (CrewAI's dependencies don't
   yet support Python 3.14, which Streamlit Cloud may default to.)
4. In **Settings -> Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxx"
   ```
5. Click Deploy (or "Reboot app").

## Notes

- Get a free Groq API key at https://console.groq.com
- The DuckDuckGo tool needs no API key — it uses the `ddgs` package.
- Groq's free tier has tight rate limits; if you hit one, wait a few seconds
  and try again.
