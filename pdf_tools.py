"""PDF and image helpers for StudyMate AI.

Pure Python (no Streamlit code), so it can be tested on its own:
    python -c "from pdf_tools import load_pdf; ..."
"""
import base64
import math
import re
from collections import Counter
from io import BytesIO

import pymupdf
from PIL import Image

# How much PDF text we send to the model per question.
# ~12,000 characters is about 3,000 tokens, which stays inside Groq free-tier limits.
MAX_CONTEXT_CHARS = 12000
MAX_OUTLINE_CHARS = 3500
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Words that tell us the student wants the WHOLE document, not one topic
BROAD_WORDS = {
    "summarize", "summarise", "summary", "overview", "all", "each", "every",
    "whole", "entire", "complete", "full", "topic", "topics", "outline",
    "pdf", "lecture", "document", "notes", "chapter", "book", "file",
    "poora", "pura", "puri", "sara", "sari", "saare", "sab", "tamam",
}

STOP_WORDS = BROAD_WORDS | {
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
    "to", "and", "or", "for", "with", "by", "from", "explain", "tell", "me",
    "about", "this", "that", "these", "those", "please", "how", "why", "when",
    "which", "who", "do", "does", "did", "can", "could", "you", "your", "i",
    "my", "it", "its", "give", "show", "describe", "define", "detail",
    "details", "simple", "simply", "easy", "words", "line", "lines", "more",
    "page", "slide", "kya", "hai", "ha", "ka", "ki", "ke", "ko", "mein",
    "ma", "me", "batao", "samjhao", "samjha", "aur", "yeh", "ye", "is",
}

VISUAL_WORDS = {
    "diagram", "diagrams", "image", "images", "picture", "pictures", "graph",
    "graphs", "chart", "charts", "figure", "figures", "photo", "visual",
    "drawing", "flowchart", "tasveer", "tasweer",
}

PAGE_PATTERN = re.compile(
    r"\b(?:page|slide|pg|safha|safa)\s*(?:no\.?|number|#)?\s*(\d{1,4})\b",
    re.IGNORECASE,
)


# ---------- Text helpers ----------

def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _stem(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text):
    """Lowercase words without stop words, roughly singular ('arrays' -> 'array')."""
    return [
        _stem(word) for word in _words(text)
        if len(word) > 1 and not word.isdigit() and word not in STOP_WORDS
    ]


def _clean(text):
    text = text.replace("�", "-")          # broken bullet characters
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _shorten(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


# ---------- Loading a PDF ----------

def load_pdf(pdf_bytes):
    """Read every page once and build everything needed to answer questions later."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    try:
        pages = []
        for page in doc:
            text = page.get_text("text").strip()

            if not text:
                # Scanned page: OCR only works if Tesseract is installed
                try:
                    text = page.get_text("text", textpage=page.get_textpage_ocr())
                except Exception:
                    text = ""

            pages.append(_clean(text))

        toc = doc.get_toc(simple=True)
    finally:
        doc.close()

    chunks = _make_chunks(pages)

    return {
        "pages": pages,                       # pages[0] is page 1
        "chunks": chunks,
        "outline": _build_outline(pages, toc),
        "index": _build_index(chunks),
    }


def _make_chunks(pages):
    chunks = []

    for page_number, text in enumerate(pages, start=1):
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append({"page": page_number, "text": text[start:end]})

            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP

    return chunks


def _build_outline(pages, toc):
    """A short map of the whole document, so the model always knows every topic."""
    if toc:
        lines = [
            f"{'  ' * (level - 1)}- {title.strip()} (page {page})"
            for level, title, page in toc
        ]
    else:
        # No bookmarks: use the first meaningful line of each page as its title
        lines = []
        previous_title = None

        for page_number, text in enumerate(pages, start=1):
            title = next(
                (line.strip() for line in text.splitlines()
                 if len(line.strip()) > 2 and not line.strip().isdigit()),
                "",
            )[:90]

            if title and title != previous_title:
                lines.append(f"- Page {page_number}: {title}")
                previous_title = title

    outline = "\n".join(lines)
    if len(outline) <= MAX_OUTLINE_CHARS:
        return outline

    # Very long books: keep an evenly spread sample of the headings
    step = math.ceil(len(outline) / MAX_OUTLINE_CHARS)
    return "\n".join(lines[::step])


# ---------- Search (BM25, the classic search-engine ranking) ----------

def _build_index(chunks):
    term_counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]

    doc_freq = Counter()
    for counts in term_counts:
        doc_freq.update(counts.keys())

    lengths = [sum(counts.values()) for counts in term_counts]
    avg_length = (sum(lengths) / len(lengths)) if lengths else 1

    return {
        "term_counts": term_counts,
        "doc_freq": doc_freq,
        "lengths": lengths,
        "avg_length": avg_length or 1,
    }


def _bm25_scores(terms, index, k1=1.5, b=0.75):
    total = len(index["term_counts"])
    scores = []

    for counts, length in zip(index["term_counts"], index["lengths"]):
        score = 0.0

        for term in terms:
            freq = counts.get(term, 0)
            if not freq:
                continue

            df = index["doc_freq"][term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            norm = k1 * (1 - b + b * length / index["avg_length"])
            score += idf * freq * (k1 + 1) / (freq + norm)

        scores.append(score)

    return scores


def mentioned_pages(question, page_count):
    """Page numbers the student named, e.g. 'explain page 12' -> [12]."""
    pages = []
    for match in PAGE_PATTERN.finditer(question):
        number = int(match.group(1))
        if 1 <= number <= page_count and number not in pages:
            pages.append(number)
    return pages[:3]


def wants_visual_answer(question):
    return bool(VISUAL_WORDS & set(_words(question)))


def best_matching_page(pdf, question):
    terms = tokenize(question)
    if not terms:
        return None

    scores = _bm25_scores(terms, pdf["index"])
    best = max(range(len(scores)), key=scores.__getitem__, default=None)

    if best is None or scores[best] <= 0:
        return None
    return pdf["chunks"][best]["page"]


def _page_overview(pages, budget):
    """A little of every page (or evenly spread pages) for whole-document questions."""
    numbered = [(n, text) for n, text in enumerate(pages, start=1) if text]
    if not numbered or budget <= 0:
        return ""

    keep = max(1, min(len(numbered), budget // 250))
    step = len(numbered) / keep
    chosen = [numbered[int(i * step)] for i in range(keep)]
    per_page = budget // keep - 15

    return "\n\n".join(
        f"[Page {n}]\n{_shorten(text, per_page)}" for n, text in chosen
    )


def _top_chunks(pdf, terms, budget):
    scores = _bm25_scores(terms, pdf["index"])
    ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

    picked = []
    used = 0
    for i in ranked:
        if scores[i] <= 0:
            break

        size = len(pdf["chunks"][i]["text"]) + 15
        if used + size > budget:
            continue

        picked.append(i)
        used += size

    if not picked:
        return _page_overview(pdf["pages"], budget)

    # Keep the document's order so the explanation flows naturally
    picked.sort()
    return "\n\n".join(
        f"[Page {pdf['chunks'][i]['page']}]\n{pdf['chunks'][i]['text']}"
        for i in picked
    )


def build_pdf_context(pdf, question, previous_question="", budget=MAX_CONTEXT_CHARS):
    """Choose which parts of the PDF to send to the model for this question."""
    pages = pdf["pages"]

    whole_document = "\n\n".join(
        f"[Page {n}]\n{text}" for n, text in enumerate(pages, start=1) if text
    )
    if len(whole_document) <= budget:
        return whole_document                 # small PDF: send everything

    parts = [f"Outline of the whole document:\n{pdf['outline']}"]
    used = len(parts[0])

    named_pages = mentioned_pages(question, len(pages))
    for n in named_pages:
        block = _shorten(f"[Page {n}]\n{pages[n - 1]}", budget // 3)
        parts.append(block)
        used += len(block)

    terms = tokenize(question)

    # Follow-ups like "explain more" reuse the previous question's topic
    if not terms and not named_pages and not (BROAD_WORDS & set(_words(question))):
        terms = tokenize(previous_question)

    remaining = budget - used
    if terms:
        parts.append(_top_chunks(pdf, terms, remaining))
    elif not named_pages:
        parts.append(_page_overview(pages, remaining))

    return "\n\n".join(part for part in parts if part)


# ---------- Images ----------

def image_to_data_url(image_bytes, max_side=1600):
    """Shrink and convert an image so it is small enough for the vision API."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_side, max_side))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


def render_page_image(pdf_bytes, page_number):
    """Take a 'screenshot' of one PDF page, so the vision model can see its diagrams."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    return image_to_data_url(png_bytes)
