"""
streamlit_app.py - Student Web Application for Cambridge IGCSE Physics Paper 2 (MCQ)
Port: 8502
Engine Version: 1.2 (Dual-Column Precision Scanner)
"""

import streamlit as st
import pandas as pd
import datetime
import mcq_engine
import config

TEACHER_PIN = getattr(config, "TEACHER_PIN", "1234")

st.set_page_config(
    page_title="IGCSE Physics MCQ Auto-Grader",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for modern look and mobile responsiveness
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.2rem 1rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        color: #dbeafe;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .score-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .score-val {
        font-size: 3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin: 0;
        line-height: 1;
    }
    .score-pct {
        font-size: 1.3rem;
        font-weight: 600;
        color: #64748b;
        margin: 0.3rem 0;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.3rem 1.2rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 20px;
        color: white;
        margin-top: 0.5rem;
    }
    .grade-astar { background-color: #059669; }
    .grade-a     { background-color: #10b981; }
    .grade-b     { background-color: #2563eb; }
    .grade-c     { background-color: #d97706; }
    .grade-lower { background-color: #ef4444; }

    .wrong-card {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 6px;
    }
    .format-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Main Application Title
st.markdown("""
<div class="main-header">
    <h1>⚡ IGCSE Physics Paper 2 Auto-Grader</h1>
    <p>Cambridge 0625 Multiple Choice (40 Marks) • Instant Score & Error Analysis</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# STUDENT SUBMISSION & GRADING INTERFACE
# ==============================================================================
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Enter Student Details")
    student_name = st.text_input(
        "Student Full Name *",
        placeholder="e.g. Retal, Abdelrahman, Jasmine...",
        help="Please enter your name so your teacher knows who submitted the exam."
    )

    st.subheader("2. Select Your Exam")
    exam_choices = mcq_engine.get_exam_choices()
    if not exam_choices:
        st.error("No answer keys loaded! Please make sure answer_keys.json is generated.")
        selected_paper_code = None
        selected_exam_name = ""
    else:
        exam_labels = [name for code, name in exam_choices]
        selected_idx = st.selectbox(
            "Choose Past Paper *",
            range(len(exam_labels)),
            format_func=lambda i: exam_labels[i],
            help="Select the exact Cambridge Paper 2 exam session and variant."
        )
        selected_paper_code = exam_choices[selected_idx][0]
        selected_exam_name = exam_choices[selected_idx][1]

    st.subheader("3. Upload Your Solved Exam")
    st.caption("Upload photos or PDF of your answers. Any of the following 3 formats is supported:")
    st.markdown("""
    - 📄 **Format 1:** Actual question paper with circled/marked options  
    - 🔘 **Format 2:** Filled bubble sheet / OMR card  
    - 📓 **Format 3:** Notebook page with handwritten answers (e.g. *1. B, 2. A, 3. C...*)
    """)

    uploaded_files = st.file_uploader(
        "Upload file(s) (PDF or JPG/PNG photos)",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="You can upload multiple photos if your answers span more than 1 page."
    )

    grade_clicked = st.button("🚀 Grade My Exam Now", type="primary", use_container_width=True)

with col_right:
    if grade_clicked:
        if not student_name.strip():
            st.warning("⚠️ Please enter your name before grading.")
        elif not selected_paper_code:
            st.warning("⚠️ Please select an exam.")
        elif not uploaded_files:
            st.warning("⚠️ Please upload your answer sheet or notebook photo.")
        else:
            with st.spinner("🔍 Reading answers with AI Vision... please wait (~5 seconds)"):
                try:
                    images = mcq_engine.convert_files_to_images(uploaded_files)
                    if not images:
                        st.error("Could not read any images from the uploaded files. Please check the file format.")
                    else:
                        # AI Vision extraction
                        extraction = mcq_engine.extract_answers_with_vision(images, exam_title=selected_exam_name)
                        extracted_answers = extraction.get("answers", {})
                        detected_format = extraction.get("detected_format", "Unknown")
                        notes = extraction.get("notes", "")

                        # Grade against official Cambridge answer key
                        result = mcq_engine.grade_submission(
                            student_answers=extracted_answers,
                            paper_code=selected_paper_code,
                            student_name=student_name.strip(),
                            detected_format=detected_format
                        )
                        result["notes"] = notes

                        # Save to submissions log for teacher
                        mcq_engine.save_submission(result)

                        # Store in session state for display
                        st.session_state["last_result"] = result
                        st.success("✅ Correction completed successfully!")

                except Exception as e:
                    st.error(f"Grading failed: {str(e)}")

    # Display result if available
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]

        # Grade badge CSS class
        g = res["grade"]
        badge_class = "grade-astar" if g == "A*" else ("grade-a" if g == "A" else ("grade-b" if g == "B" else ("grade-c" if g == "C" else "grade-lower")))

        notes_disp = res.get("notes", "Standard Scanner")

        st.markdown(f"""
        <div class="score-card">
            <div style="font-size: 1.1rem; color: #475569; font-weight: 600;">{res['student_name']}</div>
            <div style="font-size: 0.9rem; color: #64748b;">{res['exam_name']}</div>
            <p class="score-val">{res['score']} / 40</p>
            <p class="score-pct">{res['percentage']}% Score</p>
            <div class="grade-badge {badge_class}">Grade: {res['grade']}</div>
            <div style="margin-top: 0.8rem;">
                <span class="format-tag">Scanner v1.2: {notes_disp}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("✅ Correct", f"{res['correct_count']} / 40")
        m2.metric("❌ Wrong", f"{res['wrong_count']}")
        m3.metric("⚪ Blank / Skipped", f"{res['blank_count']}")

        st.divider()

        # Wrong Answers Table
        st.subheader(f"❌ Questions to Review ({res['wrong_count']})")
        if res["wrong_count"] == 0:
            st.balloons()
            st.success("🎉 Outstanding! Perfect score! All 40 answers are correct!")
        else:
            wrong_data = []
            for w in res["wrong_answers"]:
                student_disp = "Blank" if w["student_answer"] == "BLANK" else w["student_answer"]
                wrong_data.append({
                    "Question #": f"Q{w['question']}",
                    "Your Answer": student_disp,
                    "Official Mark Scheme": w["correct_answer"]
                })
            df_wrong = pd.DataFrame(wrong_data)
            st.dataframe(df_wrong, use_container_width=True, hide_index=True)

        # Full 1-40 Breakdown (Expander)
        with st.expander("📋 View Complete 40-Question Breakdown"):
            full_rows = []
            for row in res["full_table"]:
                icon = "✅ Correct" if row["is_correct"] else ("⚪ Blank" if row["status"] == "blank" else "❌ Wrong")
                if row["status"] == "discounted":
                    icon = "⭐ Free Mark"
                full_rows.append({
                    "Q#": row["question"],
                    "Your Answer": row["student_answer"],
                    "Correct Answer": row["correct_answer"],
                    "Status": icon
                })
            st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)

    else:
        st.info("👈 Upload your answers on the left and click **Grade My Exam Now** to see your results here.")


# ==============================================================================
# SECURE TEACHER PORTAL (Hidden at bottom behind password)
# ==============================================================================
st.write("")
st.write("")
st.divider()

with st.expander("🔒 Teacher Portal (Password Protected)", expanded=False):
    if "teacher_authed" not in st.session_state:
        st.session_state["teacher_authed"] = False

    if not st.session_state["teacher_authed"]:
        p_col1, p_col2 = st.columns([2, 1])
        with p_col1:
            entered_pin = st.text_input("Enter Teacher PIN", type="password", key="teacher_pin_input")
        with p_col2:
            st.write("")
            st.write("")
            if st.button("Unlock Dashboard"):
                if entered_pin == TEACHER_PIN:
                    st.session_state["teacher_authed"] = True
                    st.rerun()
                else:
                    st.error("Incorrect PIN.")
    else:
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.subheader("📊 Teacher Dashboard — All Submissions")
        with h_col2:
            if st.button("🔒 Lock Portal"):
                st.session_state["teacher_authed"] = False
                st.rerun()

        submissions = mcq_engine.load_submissions_history()

        if not submissions:
            st.info("No student submissions yet. Once students submit their exams, their scores and error lists will appear here.")
        else:
            total_subs = len(submissions)
            avg_score = round(sum(s["score"] for s in submissions) / total_subs, 1)
            avg_pct = round(sum(s["percentage"] for s in submissions) / total_subs, 1)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Submissions", total_subs)
            c2.metric("Class Average Score", f"{avg_score} / 40")
            c3.metric("Average Percentage", f"{avg_pct}%")

            st.divider()

            f_col1, f_col2 = st.columns([1, 1])
            with f_col1:
                search_query = st.text_input("🔍 Search by Student Name", "", key="teacher_search")
            with f_col2:
                all_exams = sorted(list(set(s["exam_name"] for s in submissions)))
                exam_filter = st.selectbox("Filter by Exam", ["All Exams"] + all_exams, key="teacher_exam_filter")

            filtered = submissions
            if search_query.strip():
                filtered = [s for s in filtered if search_query.strip().lower() in s["student_name"].lower()]
            if exam_filter != "All Exams":
                filtered = [s for s in filtered if s["exam_name"] == exam_filter]

            st.caption(f"Showing {len(filtered)} submission(s)")

            table_records = []
            for s in filtered:
                wrong_q_str = ", ".join(f"Q{q}" for q in s.get("wrong_questions", []))
                if not wrong_q_str:
                    wrong_q_str = "None (Full Marks)"
                table_records.append({
                    "Date & Time": s["timestamp"],
                    "Student": s["student_name"],
                    "Exam": s["exam_name"],
                    "Score": f"{s['score']} / 40",
                    "Percentage": f"{s['percentage']}%",
                    "Grade": s["grade"],
                    "Wrong Questions": wrong_q_str
                })

            df_display = pd.DataFrame(table_records)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Teacher Summary as CSV",
                data=csv_data,
                file_name=f"mcq_submissions_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
