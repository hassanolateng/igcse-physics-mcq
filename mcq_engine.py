"""
mcq_engine.py - Core Engine for Cambridge IGCSE Physics Paper 2 (MCQ) Auto-Grader
Supports:
1. Cambridge OMR / Bubble answer sheets (Dual-column high-precision parallel scanner)
2. Circled / marked exam question papers
3. Handwritten notebook paper answers (e.g., '1. A, 2. B, 3. C...')
"""

import os
import io
import json
import re
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from PIL import Image
import pymupdf
from google import genai
from google.genai import types

import config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcq_data")
KEYS_FILE = os.path.join(DATA_DIR, "answer_keys.json")
SUBMISSIONS_FILE = os.path.join(DATA_DIR, "submissions.json")

# Grade boundaries for Cambridge IGCSE Physics Paper 2 (out of 40)
GRADE_THRESHOLDS = [
    (34, "A*"),
    (29, "A"),
    (24, "B"),
    (19, "C"),
    (15, "D"),
    (11, "E"),
    (0,  "U")
]

def get_api_key() -> str:
    """Dynamically get API key from Streamlit secrets or config."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            val = str(st.secrets["GEMINI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass
    return getattr(config, "GEMINI_API_KEY", "").strip()

def load_answer_keys() -> Dict[str, Any]:
    """Load all pre-loaded Paper 2 answer keys."""
    if not os.path.exists(KEYS_FILE):
        return {}
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading answer keys: {e}")
        return {}

def get_exam_choices() -> List[Tuple[str, str]]:
    """Return sorted list of (paper_code, display_name) for UI dropdown."""
    keys = load_answer_keys()
    items = list(keys.items())
    items.sort(key=lambda x: (x[1].get("year", 0), x[1].get("paper_code", "")), reverse=True)
    return [(k, v["display_name"]) for k, v in items]

def convert_files_to_images(files) -> List[Image.Image]:
    """Convert uploaded files (PDF or images) into a list of PIL Images."""
    images = []
    for file_obj in files:
        name = getattr(file_obj, "name", "file").lower()
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
        elif isinstance(file_obj, (bytes, bytearray)):
            content = file_obj
        elif isinstance(file_obj, str) and os.path.exists(file_obj):
            with open(file_obj, "rb") as f:
                content = f.read()
            name = file_obj.lower()
        else:
            continue

        if name.endswith(".pdf"):
            doc = pymupdf.open(stream=content, filetype="pdf")
            zoom = 2.0
            mat = pymupdf.Matrix(zoom, zoom)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
            doc.close()
        else:
            try:
                img = Image.open(io.BytesIO(content))
                img = img.convert("RGB")
                max_dim = 2500
                if img.width > max_dim or img.height > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                print(f"Error reading image: {e}")

    return images

def robust_json_loads(raw_text: str) -> Dict[str, Any]:
    """Parse JSON with cleaning for markdown code blocks and backslashes."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            snippet = text[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        raise ValueError(f"Could not parse valid JSON from AI response:\n{text[:300]}...")

def _try_parallel_cambridge_bubble_sheet(client, img: Image.Image, model_name: str) -> Dict[str, Any]:
    """
    High-precision parallel scanner for Cambridge Supplementary Answer Sheets.
    Crops Left Column (Q1-20) and Right Column (Q21-40) separately to completely
    eliminate row drift and misalignments.
    """
    w, h = img.size
    col1 = img.crop((int(w * 0.30), int(h * 0.34), int(w * 0.59), int(h * 0.91)))
    col2 = img.crop((int(w * 0.59), int(h * 0.34), int(w * 0.88), int(h * 0.91)))

    p1 = """You are reading Questions 1 to 20 on a Cambridge Multiple Choice Answer Sheet.
There are 20 rows (numbered 1 to 20).
In each row, options are A, B, C, D, E, F, G, H.
Directly UNDERNEATH each letter is a small horizontal rectangular box [__].
The candidate answered by darkening the horizontal box UNDER their chosen letter (A, B, C, or D).
Ignore options E, F, G, H.
If no box is filled in a row, write "BLANK".

Return strictly a JSON object: {"1": "B", "2": "C", ... "20": "C"}"""

    p2 = """You are reading Questions 21 to 40 on a Cambridge Multiple Choice Answer Sheet.
There are 20 rows (numbered 21 to 40).
In each row, options are A, B, C, D, E, F, G, H.
Directly UNDERNEATH each letter is a small horizontal rectangular box [__].
The candidate answered by darkening the horizontal box UNDER their chosen letter (A, B, C, or D).
Ignore options E, F, G, H.
If no box is filled in a row, write "BLANK".

Return strictly a JSON object: {"21": "A", "22": "D", ... "40": "B"}"""

    def fetch_col(p, crop_img):
        res = client.models.generate_content(
            model=model_name,
            contents=[p, crop_img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        return robust_json_loads(res.text)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(fetch_col, p1, col1)
        f2 = executor.submit(fetch_col, p2, col2)
        res1 = f1.result()
        res2 = f2.result()

    merged = {**res1, **res2}

    # Validate that we got a substantial number of questions
    if len(merged) >= 30:
        cleaned_answers = {}
        for q in range(1, 41):
            val = str(merged.get(str(q), merged.get(q, "BLANK"))).strip().upper()
            if val in ["A", "B", "C", "D"]:
                cleaned_answers[str(q)] = val
            else:
                cleaned_answers[str(q)] = "BLANK"

        return {
            "detected_format": "bubble_sheet",
            "total_questions_detected": 40,
            "answers": cleaned_answers,
            "notes": "Processed via Cambridge dual-column high-precision scanner."
        }
    raise ValueError("Not enough answers found via dual-column crop.")

def extract_answers_with_vision(images: List[Image.Image], exam_title: str = "") -> Dict[str, Any]:
    """
    Send the student's answer sheet/paper images to Gemini Vision.
    Extracts answers 1 through 40.
    """
    if not images:
        raise ValueError("No valid images provided for answer extraction.")

    api_key = get_api_key()
    if not api_key:
        raise ValueError("No GEMINI_API_KEY configured. Please check config.py or Streamlit Secrets.")

    client = genai.Client(api_key=api_key)
    model_name = getattr(config, "DEFAULT_MODEL", "gemini-3.5-flash-lite")

    # Fast-path: Single portrait page (Cambridge Answer Sheet optimization)
    if len(images) == 1:
        img = images[0]
        if img.height > img.width:
            try:
                return _try_parallel_cambridge_bubble_sheet(client, img, model_name)
            except Exception as e:
                print(f"Dual-column scan fallback: {e}")

    # Standard general-purpose Vision prompt for exam papers & notebooks
    prompt = f"""You are an expert Cambridge IGCSE examination optical reader.
Your task is to examine the provided images of a student's solved exam and extract the candidate's chosen answer (A, B, C, or D) for every question from Question 1 through Question 40.

Target Exam: {exam_title if exam_title else "Cambridge IGCSE Physics 0625 Paper 2 (40 Questions)"}

THE SUBMISSION MAY BE IN ONE OF THREE FORMATS:
1. CIRCLED / MARKED QUESTION PAPER: The candidate circled, boxed, or underlined option A, B, C, or D.
2. BUBBLE / OMR SHEET: Filled bubbles or shaded boxes for options A, B, C, D.
3. HANDWRITTEN NOTEBOOK: Student wrote numbers and letters (e.g. "1. B, 2. A, 3. C...").

CRITICAL EXTRACTION RULES:
- Output an answer for EVERY question from 1 to 40.
- Output MUST strictly be single uppercase letters: "A", "B", "C", or "D" (or "BLANK" if unattempted).
- If an option was crossed out and another written/chosen, take the final intended choice.

OUTPUT FORMAT:
Return strictly a valid JSON object matching this exact structure:
{{
  "detected_format": "circled_exam | bubble_sheet | handwritten_notebook",
  "total_questions_detected": 40,
  "answers": {{
    "1": "A",
    "2": "C",
    "3": "B",
    ...
    "40": "D"
  }},
  "notes": "Brief notes"
}}
"""

    contents = [prompt]
    contents.extend(images)

    models_to_try = [
        model_name,
        "gemini-3.8-flash"
    ]

    last_err = None
    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            raw_text = response.text
            parsed = robust_json_loads(raw_text)

            raw_answers = parsed.get("answers", {})
            cleaned_answers = {}
            for q in range(1, 41):
                ans = str(raw_answers.get(str(q), raw_answers.get(q, "BLANK"))).strip().upper()
                if ans in ["A", "B", "C", "D"]:
                    cleaned_answers[str(q)] = ans
                else:
                    cleaned_answers[str(q)] = "BLANK"

            parsed["answers"] = cleaned_answers
            return parsed

        except Exception as e:
            last_err = e
            print(f"Model {m} extraction failed: {e}. Trying fallback...")

    raise RuntimeError(f"All Gemini models failed to extract answers. Error: {last_err}")

def grade_submission(
    student_answers: Dict[str, str],
    paper_code: str,
    student_name: str = "Student",
    detected_format: str = "Unknown"
) -> Dict[str, Any]:
    """Compare extracted student answers against official answer key."""
    all_keys = load_answer_keys()
    if paper_code not in all_keys:
        raise ValueError(f"Exam paper code '{paper_code}' not found in database.")

    exam_info = all_keys[paper_code]
    official_answers = exam_info["answers"]

    score = 0
    wrong_answers = []
    full_table = []
    blank_count = 0

    for q_int in range(1, 41):
        q_str = str(q_int)
        correct_ans = official_answers.get(q_str, "A").upper()
        student_ans = student_answers.get(q_str, "BLANK").upper()

        if student_ans == "BLANK" or not student_ans:
            blank_count += 1
            is_correct = False
            status = "blank"
        elif correct_ans == "DISCOUNTED":
            score += 1
            is_correct = True
            status = "discounted"
        elif student_ans == correct_ans:
            score += 1
            is_correct = True
            status = "correct"
        elif "/" in correct_ans or "OR" in correct_ans:
            accepted = [c for c in ["A", "B", "C", "D"] if c in correct_ans]
            if student_ans in accepted:
                score += 1
                is_correct = True
                status = "correct"
            else:
                is_correct = False
                status = "wrong"
        else:
            is_correct = False
            status = "wrong"

        if not is_correct and status != "discounted":
            wrong_answers.append({
                "question": q_int,
                "student_answer": student_ans,
                "correct_answer": correct_ans,
                "status": status
            })

        full_table.append({
            "question": q_int,
            "student_answer": student_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct,
            "status": status
        })

    percentage = round((score / 40.0) * 100.0, 1)

    grade = "U"
    for min_score, g in GRADE_THRESHOLDS:
        if score >= min_score:
            grade = g
            break

    result = {
        "student_name": student_name,
        "paper_code": paper_code,
        "exam_name": exam_info["display_name"],
        "syllabus_paper": exam_info.get("syllabus_paper", "0625/2"),
        "year": exam_info.get("year", 2024),
        "session": exam_info.get("session", "June"),
        "score": score,
        "total": 40,
        "percentage": percentage,
        "grade": grade,
        "correct_count": score,
        "wrong_count": len(wrong_answers),
        "blank_count": blank_count,
        "wrong_answers": wrong_answers,
        "full_table": full_table,
        "student_answers": student_answers,
        "detected_format": detected_format,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return result

def save_submission(result_data: Dict[str, Any]) -> bool:
    """Save submission record to submissions.json for teacher tracking."""
    os.makedirs(DATA_DIR, exist_ok=True)
    history = []
    if os.path.exists(SUBMISSIONS_FILE):
        try:
            with open(SUBMISSIONS_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    record = {
        "id": f"sub_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{result_data.get('student_name', 'std')}",
        "timestamp": result_data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "student_name": result_data.get("student_name", "Student"),
        "paper_code": result_data.get("paper_code", ""),
        "exam_name": result_data.get("exam_name", ""),
        "score": result_data.get("score", 0),
        "total": 40,
        "percentage": result_data.get("percentage", 0.0),
        "grade": result_data.get("grade", "U"),
        "wrong_count": result_data.get("wrong_count", 0),
        "blank_count": result_data.get("blank_count", 0),
        "detected_format": result_data.get("detected_format", "Unknown"),
        "wrong_questions": [w["question"] for w in result_data.get("wrong_answers", [])]
    }

    history.insert(0, record)

    try:
        with open(SUBMISSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving submission: {e}")
        return False

def load_submissions_history() -> List[Dict[str, Any]]:
    """Return all recorded submissions for the teacher dashboard."""
    if not os.path.exists(SUBMISSIONS_FILE):
        return []
    try:
        with open(SUBMISSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
