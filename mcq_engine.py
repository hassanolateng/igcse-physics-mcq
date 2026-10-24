"""
mcq_engine.py - Core Engine for Cambridge IGCSE Physics Paper 2 (MCQ) Auto-Grader
Supports:
1. Circled / marked exam question papers
2. Bubble sheets (OMR)
3. Handwritten notebook paper answers (e.g., '1. A, 2. B, 3. C...')
"""

import os
import io
import json
import re
import datetime
from typing import List, Dict, Any, Tuple
from PIL import Image
import pymupdf
from google import genai
from google.genai import types

# Import API configuration from existing config.py (READ-ONLY)
import config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcq_data")
KEYS_FILE = os.path.join(DATA_DIR, "answer_keys.json")
SUBMISSIONS_FILE = os.path.join(DATA_DIR, "submissions.json")

# Grade boundaries for Cambridge IGCSE Physics Paper 2 (out of 40)
# Based on historical Cambridge threshold trends
GRADE_THRESHOLDS = [
    (34, "A*"),
    (29, "A"),
    (24, "B"),
    (19, "C"),
    (15, "D"),
    (11, "E"),
    (0,  "U")
]

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
    # Sort by year descending, then session, then variant
    items = list(keys.items())
    items.sort(key=lambda x: (x[1].get("year", 0), x[1].get("paper_code", "")), reverse=True)
    return [(k, v["display_name"]) for k, v in items]

def convert_files_to_images(files) -> List[Image.Image]:
    """
    Convert uploaded files (PDF or images) into a list of PIL Images.
    Accepts uploaded file objects from Streamlit or raw bytes/paths.
    """
    images = []
    for file_obj in files:
        # Determine filename or name attribute
        name = getattr(file_obj, "name", "file").lower()
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            # Reset seek position if possible
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
            # Render PDF pages with PyMuPDF
            doc = pymupdf.open(stream=content, filetype="pdf")
            zoom = 2.0  # 144-200 DPI equivalent
            mat = pymupdf.Matrix(zoom, zoom)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
            doc.close()
        else:
            # Direct image file (PNG, JPG, JPEG, WEBP)
            try:
                img = Image.open(io.BytesIO(content))
                img = img.convert("RGB")
                # Resize if excessively large (> 2500px in either dimension) to save bandwidth
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
        # Try finding outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            snippet = text[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        raise ValueError(f"Could not parse valid JSON from AI response:\n{text[:300]}...")

def extract_answers_with_vision(images: List[Image.Image], exam_title: str = "") -> Dict[str, Any]:
    """
    Send the student's answer sheet/paper images to Gemini Vision.
    Extracts answers 1 through 40.
    """
    if not images:
        raise ValueError("No valid images provided for answer extraction.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    prompt = f"""You are an expert Cambridge IGCSE Physics Paper 2 (Multiple Choice) examination reader.
Your task is to examine the provided images of a student's solved exam and extract the candidate's chosen answer for every question from Question 1 through Question 40.

Target Exam: {exam_title if exam_title else "Cambridge IGCSE Physics 0625 Paper 2 (40 Questions)"}

THE SUBMISSION MAY BE IN ONE OF THREE FORMATS:
1. CIRCLED EXAM PAPER: The actual question paper where the candidate circled, boxed, underlined, or checked an option letter (A, B, C, or D).
2. BUBBLE / OMR SHEET: A grid of questions 1-40 with shaded or filled bubbles for options A, B, C, D.
3. HANDWRITTEN NOTEBOOK: A student notebook or paper where the student wrote numbers and letters (e.g., "1. B, 2. A, 3. C...", or two columns of answers).

CRITICAL EXTRACTION RULES:
- Extract an answer for EVERY question number from 1 to 40.
- Valid answers are strictly single uppercase letters: "A", "B", "C", or "D".
- If a question was skipped, blank, or completely omitted, write "BLANK".
- CROSSED-OUT ANSWERS: If a student crossed out or scribbled over an answer and wrote/circled a new one, IGNORE the crossed-out answer and extract their final intended choice.
- CIRCLED OPTION TEXT: If the student circled the text of an option rather than the letter, determine whether that text corresponds to option A, B, C, or D.
- MULTI-COLUMN LAYOUT: In notebooks, questions are often arranged in columns (e.g. 1-20 on left, 21-40 on right). Scan all columns carefully.

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
  "notes": "Any brief observation regarding unclear markings or cross-outs"
}}
"""

    contents = [prompt]
    contents.extend(images)

    # Models to try (primary with fallback)
    models_to_try = [
        getattr(config, "DEFAULT_MODEL", "gemini-3.5-flash-lite"),
        "gemini-2.5-flash"
    ]

    last_err = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            raw_text = response.text
            parsed = robust_json_loads(raw_text)

            # Ensure all questions 1-40 are present in the dictionary
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
            print(f"Model {model_name} extraction failed: {e}. Trying fallback...")

    raise RuntimeError(f"All Gemini models failed to extract answers. Error: {last_err}")

def grade_submission(
    student_answers: Dict[str, str],
    paper_code: str,
    student_name: str = "Student",
    detected_format: str = "Unknown"
) -> Dict[str, Any]:
    """
    Compare extracted student answers against official answer key.
    Calculates marks, percentage, grade, and identifies wrong answers.
    """
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
            # Cambridge discounted question: free mark to all candidates
            score += 1
            is_correct = True
            status = "discounted"
        elif student_ans == correct_ans:
            score += 1
            is_correct = True
            status = "correct"
        elif "/" in correct_ans or "OR" in correct_ans:
            # Cambridge accepted multiple letters
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

    # Determine grade
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

    # Keep light summary record for history list
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

    history.insert(0, record)  # Newest first

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
