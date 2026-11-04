"""Prompts and answer parsing for the Quiz Generator, Notes Summarizer and Study Planner.

No Streamlit code here, same idea as pdf_tools.py.
"""
import json
import random
import re
from datetime import date, timedelta

# Longer plans need too many tokens for the Groq free tier
MAX_PLAN_DAYS = 45

QUIZ_FORMAT = (
    '{"questions": [{"question": "...", "options": ["...", "...", "...", "..."], '
    '"answer_index": 0, "explanation": "...", "page": null}]}'
)

SUMMARY_STYLES = {
    "Key points": (
        "List the most important key points as short bullet points, "
        "grouped under clear headings."
    ),
    "Short summary": (
        "Write a short summary of 1-2 paragraphs that covers the main ideas."
    ),
    "Revision notes": (
        "Write exam revision notes: a heading for each topic, short bullet "
        "points, important definitions in bold, and a final section called "
        "'Remember for the exam' with the 5 most important facts."
    ),
    "Definitions table": (
        "Make a markdown table with columns Term | Meaning | Example for "
        "every important term in the material."
    ),
}


def quiz_prompt(topic, count, difficulty, material=""):
    if material:
        source = (
            "Base every question ONLY on the study material below."
            + (f" Focus on: {topic}." if topic else "")
            + f"\n\nStudy material:\n{material}"
        )
    else:
        source = f"Topic: {topic}"

    return (
        f"Create a multiple-choice quiz with exactly {count} questions.\n"
        f"Difficulty: {difficulty}.\n"
        "Rules:\n"
        "- Each question has exactly 4 options and only one correct answer.\n"
        "- Mix definitions, concepts, examples and small 'what happens if' questions.\n"
        "- Do not start options with letters like 'A)'.\n"
        "- explanation: 1-2 simple sentences on why the answer is correct.\n"
        "- page: the page number the question comes from, or null.\n"
        f"Return ONLY valid JSON in this format:\n{QUIZ_FORMAT}\n\n"
        f"{source}"
    )


def parse_quiz(raw_text):
    """Turn the model's JSON into a clean list of questions (with shuffled options)."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("The AI did not return a quiz. Please try again.")

    data = json.loads(match.group(0))
    questions = []

    for item in data.get("questions", []):
        options = [
            re.sub(r"^[A-Da-d][\).:]\s+", "", str(option)).strip()
            for option in item.get("options", [])
        ]
        answer = item.get("answer_index")

        # Skip broken questions instead of showing them
        if (
            not str(item.get("question", "")).strip()
            or len(options) < 2
            or len(set(options)) != len(options)
            or not isinstance(answer, int)
            or not 0 <= answer < len(options)
        ):
            continue

        # Models like to put the right answer first, so shuffle the options
        correct = options[answer]
        random.shuffle(options)

        page = item.get("page")
        questions.append({
            "question": str(item["question"]).strip(),
            "options": options,
            "answer_index": options.index(correct),
            "explanation": str(item.get("explanation", "")).strip(),
            "page": page if isinstance(page, int) else None,
        })

    if not questions:
        raise ValueError("The quiz came back empty. Please try again.")

    return questions


def summary_prompt(material, style, focus=""):
    return (
        "Summarize the study material below for a student. "
        f"{SUMMARY_STYLES[style]} "
        "Use simple language. When the material has page markers like "
        "[Page 4], cite pages like (Page 4). Do not add facts that are not "
        "in the material."
        + (f" Focus on: {focus}." if focus else "")
        + f"\n\nStudy material:\n{material}"
    )


# ---------- Study Planner ----------

PLAN_FORMAT = (
    '{"days": [{"date": "YYYY-MM-DD", "focus": "...", '
    '"tasks": ["... (45 min)", "... (30 min)"]}], "tips": ["..."]}'
)


def study_dates(exam_date, rest_on_sunday, today=None):
    """Every day from today until the day before the exam (dates are worked out
    in Python, because language models are bad at calendars)."""
    today = today or date.today()
    days = [today + timedelta(days=n) for n in range((exam_date - today).days)]

    if rest_on_sunday:
        days = [day for day in days if day.weekday() != 6]
    return days


def planner_prompt(subjects, dates, exam_date, hours, weak_topics=""):
    date_list = "\n".join(day.strftime("%Y-%m-%d (%A)") for day in dates)

    return (
        f"Make a day-by-day study plan for an exam on "
        f"{exam_date.strftime('%A, %d %B %Y')}.\n"
        f"Study time per day: {hours} hours.\n"
        f"Subjects and topics to cover:\n{subjects}\n"
        + (f"Give extra time to these weak topics: {weak_topics}\n" if weak_topics else "")
        + "Rules:\n"
        f"- Make exactly one entry for each of these dates, and no other dates:\n{date_list}\n"
        "- Cover every topic before the last 2 days; use the last days for "
        "revision and practice questions.\n"
        "- Each day: a short focus title and 2-3 tasks that fit in the study "
        "time, with minutes like '(45 min)'.\n"
        "- Mix learning with practice (MCQs, past papers, making notes).\n"
        "- tips: 3-5 short study tips.\n"
        f"Return ONLY valid JSON in this format:\n{PLAN_FORMAT}"
    )


def parse_plan(raw_text, dates):
    """Keep only the planned days that match our real study dates."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("The AI did not return a plan. Please try again.")

    data = json.loads(match.group(0))
    wanted = {day.isoformat() for day in dates}
    days = {}

    for item in data.get("days", []):
        day = str(item.get("date", "")).strip()[:10]
        tasks = [str(task).strip() for task in item.get("tasks", []) if str(task).strip()]

        if day in wanted and day not in days and tasks:
            days[day] = {
                "date": day,
                "focus": str(item.get("focus", "")).strip() or "Study",
                "tasks": tasks,
            }

    if not days:
        raise ValueError("The plan came back empty. Please try again.")

    return {
        "days": [days[day] for day in sorted(days)],
        "tips": [str(tip).strip() for tip in data.get("tips", []) if str(tip).strip()],
    }


def plan_to_markdown(plan):
    """A printable checklist version of the plan for the download button."""
    lines = [f"# Study plan (exam on {plan['exam_date'].strftime('%A, %d %B %Y')})", ""]

    for day in plan["days"]:
        day_date = date.fromisoformat(day["date"])
        lines.append(f"## {day_date.strftime('%a %d %b')}: {day['focus']}")
        lines.extend(f"- [ ] {task}" for task in day["tasks"])
        lines.append("")

    if plan["tips"]:
        lines.append("## Study tips")
        lines.extend(f"- {tip}" for tip in plan["tips"])

    return "\n".join(lines)
