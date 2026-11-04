import streamlit as st
from groq import Groq
import html
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import uuid  
from tavily import TavilyClient
from zoneinfo import ZoneInfo

from pdf_tools import (
    MAX_CONTEXT_CHARS,
    best_matching_page,
    build_pdf_context,
    image_to_data_url,
    load_pdf,
    mentioned_pages,
    render_page_image,
    wants_visual_answer,
)

from study_tools import (
    MAX_PLAN_DAYS,
    SUMMARY_STYLES,
    parse_plan,
    parse_quiz,
    plan_to_markdown,
    planner_prompt,
    quiz_prompt,
    study_dates,
    summary_prompt,
)


# groq api
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
# Online, all users share one database, so chats are only saved on your own computer
SAVE_HISTORY = st.secrets.get("SAVE_HISTORY", True)
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
TEXT_MODEL = "openai/gpt-oss-120b"
# Must be a Groq model that can see images.
# If you get "model not found", pick a current one from console.groq.com/docs/vision
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

# log json
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "llm_logs.jsonl"

def write_json_log(data):
    with LOG_FILE.open("a",encoding="utf-8") as file:
        file.write(
            json.dumps(data,ensure_ascii=False) + "\n"
        )

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SQLite Database ----------

DB_PATH = Path(__file__).parent / "study_assistant.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Add conversation_id to an existing database safely
        columns = {
            row[1] for row in
            conn.execute("PRAGMA table_info(messages)").fetchall()
        }

        if "conversation_id" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN conversation_id TEXT"
            )

        # Preserve and group existing messages into one old conversation
        old_messages = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id IS NULL"
        ).fetchone()[0]

        if old_messages:
            now = datetime.now(timezone.utc).isoformat()
            old_id = str(uuid.uuid4())

            conn.execute("""
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (old_id, "Previous Conversation", now, now))

            conn.execute("""
                UPDATE messages
                SET conversation_id = ?
                WHERE conversation_id IS NULL
            """, (old_id,))


def create_conversation(title):
    conversation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, title[:50], now, now))

    return conversation_id


def list_conversations():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT id, title
            FROM conversations
            ORDER BY updated_at DESC
        """).fetchall()


def load_messages(conversation_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY id
        """, (conversation_id,)).fetchall()

    return [{"role": role, "content": content} for role, content in rows]


def save_message(conversation_id, role, content):
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO messages
                (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, role, content, now))

        conn.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
        """, (now, conversation_id))


def delete_conversation(conversation_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,)
        )


init_db()


# ---------- Custom Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FC;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #253047;
}

[data-testid="stSidebar"] * {
    color: #F3F4F6;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    color: #FFFFFF;
    padding: 8px 0 2px;
}

.brand span {
    color: #A5B4FC;
}

.sidebar-note {
    font-size: 13px;
    color: #A5B4C8;
    margin-bottom: 28px;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172033;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.subtitle {
    color: #687386;
    font-size: 15px;
    margin-bottom: 24px;
}

.hero {
    background: linear-gradient(120deg, #312E81, #4F46E5, #6366F1);
    padding: 28px;
    border-radius: 20px;
    color: white;
    margin: 12px 0 26px;
}

.hero h2 {
    color: white;
    font-size: 25px;
    margin: 0 0 8px;
}

.hero p {
    color: #E0E7FF;
    font-size: 14px;
    margin: 0;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #172033;
    margin: 10px 0 14px;
}

.feature-card {
    background: white;
    border: 1px solid #E6EAF2;
    border-radius: 16px;
    padding: 19px 17px;
    min-height: 145px;
    margin-bottom: 10px;
}

.feature-icon {
    font-size: 26px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 15px;
    font-weight: 700;
    color: #202B40;
    margin-bottom: 6px;
}

.feature-desc {
    font-size: 12px;
    line-height: 1.6;
    color: #718096;
}

.stButton > button {
    border-radius: 11px;
    font-weight: 600;
    min-height: 43px;
    border: 1px solid #E1E6F0;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #818CF8;
    color: #4338CA;
}

/* Black text for user and AI messages, including headings and tables */
.stChatMessage [data-testid="stMarkdownContainer"],
.stChatMessage [data-testid="stMarkdownContainer"] p,
.stChatMessage [data-testid="stMarkdownContainer"] h1,
.stChatMessage [data-testid="stMarkdownContainer"] h2,
.stChatMessage [data-testid="stMarkdownContainer"] h3,
.stChatMessage [data-testid="stMarkdownContainer"] li,
.stChatMessage [data-testid="stMarkdownContainer"] span,
.stChatMessage [data-testid="stMarkdownContainer"] strong,
.stChatMessage [data-testid="stMarkdownContainer"] th,
.stChatMessage [data-testid="stMarkdownContainer"] td {
    color: #000000 !important;
}

/* Table styling */
.stChatMessage [data-testid="stMarkdownContainer"] table {
    color: #000000 !important;
    background-color: #FFFFFF !important;
}

.stChatMessage [data-testid="stMarkdownContainer"] th,
.stChatMessage [data-testid="stMarkdownContainer"] td {
    background-color: #FFFFFF !important;
    border-color: #D1D5DB !important;
}

div[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #E8EBF3;
    border-radius: 15px;
    padding: 14px;
    margin-bottom: 12px;
}

div[data-testid="stChatInput"] {
    border-color: #D9DEEA;
    border-radius: 16px;
}
/* Study tools: dark text on the light page */
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stRadio"] label p,
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[class*="st-key-tool-"] [data-testid="stMarkdownContainer"] * {
    color: #172033 !important;
}

/* ...but keep the theme's own text color on buttons */
[class*="st-key-tool-"] button [data-testid="stMarkdownContainer"] * {
    color: inherit !important;
}

[class*="st-key-tool-"] button[kind="secondary"] {
    background: #FFFFFF;
    color: #172033;
}

.quiz-question {
    font-size: 16px;
    font-weight: 700;
    color: #172033;
    margin: 16px 0 2px;
}

/* One white card per day in the Study Planner */
[class*="st-key-plan-day-"] {
    background: #FFFFFF;
    border: 1px solid #E6EAF2;
    border-radius: 14px;
    padding: 14px 18px;
}

.quiz-score {
    font-size: 24px;
    font-weight: 800;
    color: #4338CA;
    margin: 18px 0 6px;
}

.footer-note {
    color: #8992A3;
    font-size: 12px;
    text-align: center;
    padding: 16px 0;
}

@media (max-width: 640px) {
    .main-title { font-size: 27px; }
    .hero { padding: 21px; }
}
</style>
""", unsafe_allow_html=True)


# ---------- Session State(APP Ki Memory) ----------

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "study_mode" not in st.session_state:
    st.session_state.study_mode = "AI Tutor"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# The PDF or image the student attached last: {"kind": "pdf" | "image", ...}
if "active_file" not in st.session_state:
    st.session_state.active_file = None


# ---------- LLM Helpers ----------

PDF_INSTRUCTIONS = (
    "The student uploaded a document named '{name}' ({pages} pages). "
    "Below is the material from it that matters for this question: "
    "an outline of the whole document and the most relevant pages. "
    "Answer from this material and cite pages like (Page 12). "
    "When the student asks about all topics, cover every topic in the outline. "
    "Speak as someone who has read the document; do not talk about "
    "'excerpts'. Only if something is truly missing, say so in one line.\n\n"
    "{context}"
)

DEFAULT_PROMPTS = {
    "pdf": "Give me an overview of this document and explain its main topics.",
    "image": "Explain this image in detail.",
}


def call_llm(model, messages, max_tokens=2048, temperature=0.5):
    start_time = time.perf_counter()
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "llm_request",
        "provider": "groq",
        "model": model,
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        write_json_log({
            **log,
            "level": "ERROR",
            "status": "failed",
            "response_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        raise

    usage = response.usage
    write_json_log({
        **log,
        "level": "INFO",
        "status": "success",
        "response_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    })

    return response.choices[0].message.content


def system_prompt():
    now = datetime.now(ZoneInfo("Asia/Karachi"))
    return (
        "You are AI Study Assistant, a helpful study assistant. "
        "Explain subjects in simple language with examples. "
        f"Student level: {level}. Reply in: {language}. "
        f"Current date and time in Pakistan: {now.strftime('%A, %d %B %Y, %I:%M %p')}."
    )


def recent_history(limit=6, max_chars=1000):
    """Last few messages; older ones are shortened to save tokens."""
    history = st.session_state.messages[-limit:]
    older = [
        {"role": m["role"], "content": m["content"][:max_chars]}
        for m in history[:-1]
    ]
    return older + history[-1:]


def previous_user_question():
    for message in reversed(st.session_state.messages[:-1]):
        if message["role"] == "user":
            return message["content"]
    return ""


def ask_about_image(question, image_data_url, page_text=""):
    instructions = (
        system_prompt()
        + " Read all visible text and explain any diagrams, charts, graphs,"
        " tables or images step by step."
    )
    if page_text:
        instructions += f"\n\nText extracted from this page:\n{page_text[:3000]}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{instructions}\n\nQuestion: {question}"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }]
    return call_llm(VISION_MODEL, messages, max_tokens=1500, temperature=0.2)

# web search ka lia 
def web_search_results(query):
    """Search the web with Tavily and return the top results as text."""
    results = tavily.search(query=query, max_results=5)["results"]
    return "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['content']}"
        for i, r in enumerate(results, start=1)
    )

def answer_question(prompt):
    active = st.session_state.active_file

    if active and active["kind"] == "image":
        return ask_about_image(prompt, active["data_url"])
        
    if web_search:
        with st.spinner("Searching the web..."):
            results = web_search_results(prompt)

        messages = [{
            "role": "system",
            "content": (
                system_prompt()
                + " Answer using the web search results below. "
                "Cite sources like [1], [2] and list their links at the end. "
                "Rules: if a location is needed but not given, ask the user "
                "for it. Ignore results with old dates. If the results do "
                "not contain the answer, say so clearly instead of guessing.\n\n"
                + results
            ),
        }]
        messages.extend(recent_history())
        return call_llm(TEXT_MODEL, messages, max_tokens=1500)

    messages = [{"role": "system", "content": system_prompt()}]


    if active and active["kind"] == "pdf":
        pdf = active["pdf"]

        # "Explain the diagram on page 12" -> show that page to the vision model
        if wants_visual_answer(prompt):
            pages = mentioned_pages(prompt, len(pdf["pages"]))
            page = pages[0] if pages else best_matching_page(pdf, prompt)

            if page:
                st.caption(f" Looking at page {page} of {active['name']}")
                return ask_about_image(
                    prompt,
                    render_page_image(active["bytes"], page),
                    pdf["pages"][page - 1],
                )

        context = build_pdf_context(pdf, prompt, previous_user_question())
        messages.append({
            "role": "system",
            "content": PDF_INSTRUCTIONS.format(
                name=active["name"],
                pages=len(pdf["pages"]),
                context=context,
            ),
            
        })

    messages.extend(recent_history())
    return call_llm(TEXT_MODEL, messages)


def attach_file(uploaded):
    """Read a PDF or image from the + button and make it the active file."""
    data = uploaded.getvalue()

    try:
        if uploaded.name.lower().endswith(".pdf"):
            with st.spinner("Reading your PDF..."):
                pdf = load_pdf(data)

            if not any(pdf["pages"]):
                st.warning(
                    "No readable text found in this PDF. If it is scanned, "
                    "upload its pages as images instead."
                )
                return False

            st.session_state.active_file = {
                "kind": "pdf",
                "name": uploaded.name,
                "bytes": data,
                "pdf": pdf,
            }
            st.toast(f" {uploaded.name} loaded ({len(pdf['pages'])} pages)")
        else:
            st.session_state.active_file = {
                "kind": "image",
                "name": uploaded.name,
                "data_url": image_to_data_url(data),
            }
            st.toast(f" {uploaded.name} attached")

        return True

    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {type(exc).__name__}: {exc}")
        return False

# ---------- Study Tools ----------

def pdf_picker(key):
    """Use the PDF the student already attached, or let them upload one here."""
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key=key)

    # file_uploader keeps its file on every rerun, so read each file only once
    if uploaded and uploaded.file_id != st.session_state.get("last_upload_id"):
        st.session_state.last_upload_id = uploaded.file_id
        attach_file(uploaded)

    active = st.session_state.active_file
    if active and active["kind"] == "pdf":
        st.caption(f" Using: {active['name']} ({len(active['pdf']['pages'])} pages)")
        return active
    return None


def quiz_tool():
    st.markdown(
        '<div class="section-title"> Quiz Generator</div>',
        unsafe_allow_html=True
    )

    source = st.radio(
        "Make a quiz from", ["A topic", "My PDF"],
        horizontal=True, key="quiz_source"
    )

    pdf_file = None
    if source == "My PDF":
        pdf_file = pdf_picker("quiz_pdf")
        topic = st.text_input(
            "Focus on (optional)",
            placeholder="e.g. binary search, or leave empty for the whole PDF"
        )
    else:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. Python functions, photosynthesis, linked lists"
        )

    count_col, difficulty_col = st.columns(2)
    count = count_col.selectbox("Number of questions", [3, 5, 10], index=1)
    difficulty = difficulty_col.selectbox(
        "Difficulty", ["Easy", "Medium", "Hard"], index=1
    )

    if st.button(" Generate quiz", type="primary"):
        if source == "My PDF" and not pdf_file:
            st.warning("Please upload a PDF first.")
        elif source == "A topic" and not topic.strip():
            st.warning("Please type a topic.")
        else:
            material = build_pdf_context(pdf_file["pdf"], topic) if pdf_file else ""
            messages = [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": quiz_prompt(
                    topic.strip(), count, difficulty, material
                )},
            ]

            with st.spinner("Writing your quiz..."):
                try:
                    raw = call_llm(TEXT_MODEL, messages, max_tokens=3000, temperature=0.4)
                    st.session_state.quiz = {
                        "id": uuid.uuid4().hex,
                        "questions": parse_quiz(raw),
                    }
                except Exception as exc:
                    st.error(f"Could not create the quiz: {exc}")

    if st.session_state.get("quiz"):
        show_quiz(st.session_state.quiz)


def show_quiz(quiz):
    questions = quiz["questions"]

    with st.form(f"quiz_form_{quiz['id']}"):
        for number, question in enumerate(questions, start=1):
            st.markdown(
                f'<div class="quiz-question">Q{number}. '
                f'{html.escape(question["question"])}</div>',
                unsafe_allow_html=True
            )
            st.radio(
                f"Question {number}",
                question["options"],
                index=None,
                key=f"{quiz['id']}_{number}",
                label_visibility="collapsed",
            )

        if st.form_submit_button(" Submit answers", type="primary"):
            quiz["answers"] = [
                st.session_state.get(f"{quiz['id']}_{number}")
                for number in range(1, len(questions) + 1)
            ]

    if "answers" not in quiz:
        return

    # ---- Results ----
    chosen = quiz["answers"]
    score = sum(
        choice == question["options"][question["answer_index"]]
        for choice, question in zip(chosen, questions)
    )
    percent = round(score * 100 / len(questions))

    st.markdown(
        f'<div class="quiz-score">Your score: {score}/{len(questions)} ({percent}%)</div>',
        unsafe_allow_html=True
    )
    if percent >= 80:
        st.markdown(" Excellent work!")
    elif percent >= 50:
        st.markdown(" Good effort! Read the explanations for the ones you missed.")
    else:
        st.markdown(" Keep practising. The explanations below will help.")

    for number, (choice, question) in enumerate(zip(chosen, questions), start=1):
        correct = question["options"][question["answer_index"]]
        page = f" (Page {question['page']})" if question["page"] else ""

        if choice == correct:
            st.success(f"**Q{number}: Correct!** {correct}\n\n{question['explanation']}{page}")
        else:
            st.error(
                f"**Q{number}:** You chose *{choice or 'nothing'}*. "
                f"Correct answer: **{correct}**\n\n{question['explanation']}{page}"
            )

    if st.button(" Make a new quiz"):
        st.session_state.quiz = None
        st.rerun()


def summarizer_tool():
    st.markdown(
        '<div class="section-title"> Notes Summarizer</div>',
        unsafe_allow_html=True
    )

    source = st.radio(
        "Summarize", ["Pasted notes", "My PDF"],
        horizontal=True, key="summary_source"
    )

    pdf_file = None
    notes = ""
    focus = ""
    if source == "My PDF":
        pdf_file = pdf_picker("summary_pdf")
        focus = st.text_input(
            "Focus on (optional)",
            placeholder="e.g. arrays, or leave empty for the whole PDF"
        )
    else:
        notes = st.text_area(
            "Paste your notes",
            height=220,
            placeholder="Paste your lecture notes or any study text here..."
        ).strip()
        if len(notes) > MAX_CONTEXT_CHARS:
            st.warning(
                "Your notes are long, so only the first part will be summarized. "
                "Upload them as a PDF to cover everything."
            )

    style = st.selectbox("Summary style", list(SUMMARY_STYLES))

    if st.button(" Summarize", type="primary"):
        if pdf_file:
            material = build_pdf_context(pdf_file["pdf"], focus)
            name = pdf_file["name"]
        else:
            material = notes[:MAX_CONTEXT_CHARS]
            name = "my-notes"

        if not material:
            st.warning("Please paste some notes or upload a PDF first.")
        else:
            messages = [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": summary_prompt(material, style, focus)},
            ]

            with st.spinner("Summarizing..."):
                try:
                    st.session_state.summary = {
                        "name": name,
                        "style": style,
                        "text": call_llm(TEXT_MODEL, messages, max_tokens=2500, temperature=0.3),
                    }
                except Exception as exc:
                    st.error(f"Could not summarize: {exc}")

    summary = st.session_state.get("summary")
    if summary:
        with st.container(border=True):
            st.markdown(summary["text"])

        st.download_button(
            "⬇ Download notes",
            summary["text"],
            file_name=f"{Path(summary['name']).stem} - {summary['style']}.md",
            mime="text/markdown",
        )


def planner_tool():
    st.markdown(
        '<div class="section-title"> Study Planner</div>',
        unsafe_allow_html=True
    )

    source = st.radio(
        "Plan for", ["My subjects", "My PDF"],
        horizontal=True, key="plan_source"
    )

    if source == "My PDF":
        pdf_file = pdf_picker("plan_pdf")
        subjects = pdf_file["pdf"]["outline"] if pdf_file else ""
    else:
        subjects = st.text_area(
            "Subjects or topics (one per line)",
            height=140,
            placeholder=(
                "Data Structures: arrays, linked lists, stacks\n"
                "Calculus: limits, derivatives\n"
                "English: essay writing"
            )
        ).strip()

    today = date.today()
    date_col, hours_col = st.columns(2)
    exam_date = date_col.date_input(
        "Exam date",
        value=today + timedelta(days=14),
        min_value=today + timedelta(days=1),
        max_value=today + timedelta(days=MAX_PLAN_DAYS),
        format="DD/MM/YYYY",
    )
    hours = hours_col.selectbox("Study hours per day", [1, 2, 3, 4, 5, 6], index=1)

    weak_topics = st.text_input(
        "Weak topics (optional)",
        placeholder="e.g. recursion, time complexity"
    ).strip()
    rest_on_sunday = st.checkbox("Keep Sunday as a rest day", value=True)

    if st.button(" Make my plan", type="primary"):
        dates = study_dates(exam_date, rest_on_sunday)

        if not subjects:
            st.warning("Please type your subjects or upload a PDF first.")
        elif not dates:
            st.warning("There are no study days left before this exam date.")
        else:
            messages = [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": planner_prompt(
                    subjects, dates, exam_date, hours, weak_topics
                )},
            ]

            with st.spinner("Planning your study days..."):
                try:
                    raw = call_llm(TEXT_MODEL, messages, max_tokens=5000, temperature=0.4)
                    st.session_state.plan = {
                        "id": uuid.uuid4().hex,
                        "exam_date": exam_date,
                        **parse_plan(raw, dates),
                    }
                except Exception as exc:
                    st.error(f"Could not make the plan: {exc}")

    if st.session_state.get("plan"):
        show_plan(st.session_state.plan)


def show_plan(plan):
    days = plan["days"]
    done = sum(
        st.session_state.get(f"{plan['id']}_{day['date']}", False)
        for day in days
    )

    st.markdown(
        f'<div class="quiz-score">{done}/{len(days)} study days done</div>',
        unsafe_allow_html=True
    )
    st.progress(done / len(days))
    st.markdown(f" **Exam day:** {plan['exam_date'].strftime('%A, %d %B %Y')}")

    for day in days:
        day_date = date.fromisoformat(day["date"])

        with st.container(key=f"plan-day-{day['date']}"):
            st.checkbox(
                f"**{day_date.strftime('%a %d %b')}** · {day['focus']}",
                key=f"{plan['id']}_{day['date']}",
            )
            st.markdown("\n".join(f"- {task}" for task in day["tasks"]))

    if plan["tips"]:
        st.markdown(
            "** Study tips**\n" + "\n".join(f"- {tip}" for tip in plan["tips"])
        )

    st.download_button(
        "⬇ Download plan",
        plan_to_markdown(plan),
        file_name=f"Study plan - {plan['exam_date'].isoformat()}.md",
        mime="text/markdown",
    )

    if st.button(" Make a new plan"):
        st.session_state.plan = None
        st.rerun()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        '<div class="brand">AI Study<span> Assistant</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-note">Your personal AI learning companion</div>',
        unsafe_allow_html=True
    )

    st.markdown("###  Learning Space")

    mode = st.selectbox(
        "Choose a study tool",
        [
            "AI Tutor",
            "Quiz Generator",
            "Notes Summarizer",
            "Study Planner",
        ],
        key="mode_selector"
    )
    st.session_state.study_mode = mode


    st.markdown("---")
    st.markdown("###  Recent Chats")

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.active_conversation_id = None
        st.session_state.messages = []
        st.session_state.active_file = None
        st.rerun()
    if SAVE_HISTORY:
        for chat_id, chat_title in list_conversations():
            if st.button(
             f" {chat_title}",
             key=f"chat_{chat_id}",
             use_container_width=True
               ):
                st.session_state.active_conversation_id = chat_id
                st.session_state.messages = load_messages(chat_id)
                st.rerun()

    st.markdown("---")
    st.markdown("###  Preferences")

    level = st.selectbox(
        "Learning level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    language = st.selectbox(
        "Response language",
        ["Simple English", "Roman Urdu", "Urdu"]
    )

    web_search = st.toggle(
         " Web search (latest info)",
        help="Turn on for news, dates and anything recent."
    )

    active_file = st.session_state.active_file
    if active_file:
        st.markdown("---")
        icon = "" if active_file["kind"] == "pdf" else ""
        st.info(f"{icon} Active file: {active_file['name']}")
        if st.button(" Remove file", use_container_width=True):
            st.session_state.active_file = None
            st.rerun()

    st.markdown("---")

    if st.button(" Clear conversation"):
        chat_id = st.session_state.active_conversation_id

        if chat_id:
            delete_conversation(chat_id)
        st.session_state.active_conversation_id = None
        st.session_state.messages = []
        st.session_state.active_file = None
        st.rerun()

    st.markdown("---")
    st.caption(" Learn at your own pace.")


# ---------- Main Header ----------
st.markdown(
    '<div class="main-title">Your AI Study Assistant</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Understand concepts, practice questions, and study smarter.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <h2>Make learning easier. </h2>
    <p>
        Ask questions, explore difficult topics, and build your
        knowledge one step at a time.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Study Tool Screens ----------
FOOTER = (
    '<div class="footer-note">'
    'AI Study Assistant · Keep learning, keep growing '
    '</div>'
)

if mode in ("Quiz Generator", "Notes Summarizer", "Study Planner"):
    if mode == "Quiz Generator":
        with st.container(key="tool-quiz"):
            quiz_tool()
    elif mode == "Notes Summarizer":
        with st.container(key="tool-summary"):
            summarizer_tool()
    else:
        with st.container(key="tool-plan"):
            planner_tool()

    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()   # tool screens don't show the chat below

# ---------- Feature Cards ----------
if not st.session_state.messages:
    st.markdown(
        '<div class="section-title">What would you like to do today?</div>',
        unsafe_allow_html=True
    )

    feature_cols = st.columns(3)

    features = [
        ("💬", "Understand a topic", "Learn difficult concepts with simple explanations."),
        ("📄", "Ask your PDF", "Click + to attach notes or a lecture PDF and ask anything."),
        ("🖼️", "Explain a diagram", "Click + to attach a graph or diagram image.")
    ]

    for col, (icon, title, desc) in zip(feature_cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------- Chat History ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- Chat Input (+ attaches a PDF or an image) ----------
chat_value = st.chat_input(
    "Ask a question, or click + to attach a PDF or image...",
    accept_file=True,
    file_type=["pdf", *IMAGE_TYPES]
)

prompt = None
display_prompt = None
attached_image = None

if chat_value:
    prompt = (chat_value.text or "").strip()
    display_prompt = prompt

    if chat_value.files:
        uploaded = chat_value.files[0]

        if attach_file(uploaded):
            kind = st.session_state.active_file["kind"]
            if not prompt:
                prompt = DEFAULT_PROMPTS[kind]

            icon = "📄" if kind == "pdf" else "🖼️"
            display_prompt = f"{icon} **{uploaded.name}**\n\n{prompt}"

            if kind == "image":
                attached_image = uploaded.getvalue()

elif st.session_state.pending_prompt:
    prompt = display_prompt = st.session_state.pending_prompt

if prompt:
    st.session_state.pending_prompt = None
    if SAVE_HISTORY and not st.session_state.active_conversation_id:
        st.session_state.active_conversation_id = create_conversation(
            prompt[:50] or "New Conversation"
        )
    chat_id = st.session_state.active_conversation_id

    st.session_state.messages.append({
        "role": "user",
        "content": display_prompt
    })
    if SAVE_HISTORY:                                   # ← change 1
        save_message(chat_id, "user", display_prompt)

    with st.chat_message("user"):
        st.markdown(display_prompt)
        if attached_image:
            st.image(attached_image, width=320)

    with st.chat_message("assistant"):
        with st.spinner("Preparing your study response..."):
            try:
                answer = answer_question(prompt)
            except Exception as exc:
                st.error(f"AI Error: {type(exc).__name__}: {exc}")
                answer = "Sorry, the AI service failed. Please try again."
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    if SAVE_HISTORY:                                   # ← change 2
        save_message(chat_id, "assistant", answer)


st.markdown(FOOTER, unsafe_allow_html=True)
