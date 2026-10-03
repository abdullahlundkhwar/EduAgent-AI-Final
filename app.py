
import streamlit as st
from google import genai
from pypdf import PdfReader
import time


# ==============================
# Gemini API
# ==============================

api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)


# ==============================
# Retry Function
# ==============================

def generate_with_retry(prompt, max_retries=3):

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            return response.text

        except Exception as e:

            if "503" in str(e) or "UNAVAILABLE" in str(e):

                time.sleep(5)

            else:
                raise e

    raise Exception(
        "Gemini is still unavailable after several attempts."
    )


# ==============================
# PDF Text Extraction
# ==============================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==============================
# Planning Agent
# ==============================

def planning_agent(
    subject,
    grade,
    topic,
    duration,
    difficulty
):

    prompt = f"""
You are the Planning Agent of EduAgent AI,
a multi-agent teaching assistant for teachers.

Create a structured lesson plan using the information below.

Subject: {subject}
Grade: {grade}
Topic: {topic}
Class Duration: {duration} minutes
Difficulty Level: {difficulty}

Create:

1. Three clear learning objectives
2. A short introduction to the topic
3. A lesson sequence with time allocation
4. Teaching/learning activities
5. A suitable classroom activity
6. A short conclusion
7. Materials/resources required

Make the lesson appropriate for the specified grade.
Use simple, clear language.
Focus on accurate educational content.

Return the answer with clear headings.
"""

    return generate_with_retry(prompt)


# ==============================
# Content Agent
# ==============================

def content_agent(
    subject,
    grade,
    topic,
    lesson_plan
):

    prompt = f"""
You are the Content Agent of EduAgent AI.

Your job is to create high-quality teaching content
based on the lesson plan prepared by the Planning Agent.

Teacher information:
Subject: {subject}
Grade: {grade}
Topic: {topic}

Lesson plan from Planning Agent:
{lesson_plan}

Create:

1. A clear explanation of the topic
2. Important concepts and definitions
3. Important formulas, if applicable
4. One or two simple real-life examples
5. A worked example suitable for the students
6. A classroom activity
7. Three questions the teacher can ask students

Requirements:
- Keep the content appropriate for the specified grade.
- Use simple language.
- For Physics, make sure formulas and scientific concepts are accurate.
- Do not introduce advanced concepts unnecessarily.
- Organize the answer with clear headings.
"""

    return generate_with_retry(prompt)


# ==============================
# Assessment Agent
# ==============================

def assessment_agent(
    subject,
    grade,
    topic,
    content
):

    prompt = f"""
You are the Assessment Agent of EduAgent AI.

Create assessments based on the teaching content.

Subject: {subject}
Grade: {grade}
Topic: {topic}

Teaching content:
{content}

Create:

1. Five multiple-choice questions (MCQs)
   - Give 4 options for each question
   - Clearly identify the correct answer

2. Three short-answer questions
   - Suitable for the specified grade
   - Include a brief answer/key point for each

3. Two homework questions
   - One basic question
   - One application-based question

Requirements:
- Questions must be based on the provided content.
- Keep the difficulty appropriate for the grade.
- Avoid ambiguous questions.
- Make sure the answers are scientifically accurate.
- Use clear formatting and headings.
"""

    return generate_with_retry(prompt)


# ==============================
# Review Agent
# ==============================

def review_agent(
    subject,
    grade,
    topic,
    lesson_plan,
    content,
    assessment
):

    prompt = f"""
You are the Review Agent of EduAgent AI.

Review the complete teaching package.

Subject: {subject}
Grade: {grade}
Topic: {topic}

LESSON PLAN:
{lesson_plan}

TEACHING CONTENT:
{content}

ASSESSMENT:
{assessment}

Review for:

1. Scientific accuracy
2. Grade appropriateness
3. Alignment between learning objectives and content
4. Alignment between content and assessment
5. Clarity and readability
6. Correct formulas and calculations
7. Quality of MCQs and answers
8. Suitable difficulty level

Provide:

A. Overall quality rating out of 10

B. Problems or issues found

C. Specific corrections needed

D. Final recommendations for the teacher

If everything is correct, clearly say:
"No major issues found."

Keep the review concise and useful.
"""

    return generate_with_retry(prompt)


# ==============================
# Complete Multi-Agent Pipeline
# ==============================

def generate_teaching_package(
    subject,
    grade,
    topic,
    duration,
    difficulty
):

    lesson_plan = planning_agent(
        subject,
        grade,
        topic,
        duration,
        difficulty
    )

    content = content_agent(
        subject,
        grade,
        topic,
        lesson_plan
    )

    assessment = assessment_agent(
        subject,
        grade,
        topic,
        content
    )

    review = review_agent(
        subject,
        grade,
        topic,
        lesson_plan,
        content,
        assessment
    )

    return {
        "lesson_plan": lesson_plan,
        "content": content,
        "assessment": assessment,
        "review": review
    }


# ==============================
# Streamlit Interface
# ==============================

st.set_page_config(
    page_title="EduAgent AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduAgent AI")
st.subheader("Multi-Agent Teaching Assistant")

st.write(
    "Generate a complete teaching package using "
    "four specialized AI agents."
)


# Teacher Inputs

# ==============================
# Visual Styling
# ==============================

st.markdown("""
<style>
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef5ff 0%, #f7f9fc 100%);
    border: 1px solid #dbe5f1;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 750;
    margin: 0;
}
.hero-subtitle {
    font-size: 1.05rem;
    margin-top: 0.35rem;
    color: #52606d;
}
.agent-card {
    padding: 0.9rem;
    border-radius: 12px;
    border: 1px solid #e1e7ef;
    background: #ffffff;
    text-align: center;
    min-height: 105px;
}
.agent-icon { font-size: 1.6rem; }
.agent-name { font-weight: 650; margin-top: 0.25rem; }
.agent-desc { font-size: 0.82rem; color: #697586; }
.section-note { color: #667085; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ==============================
# Hero Header
# ==============================

st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 EduAgent AI</div>
    <div class="hero-subtitle">
        Multi-Agent Teaching Assistant — create a complete, classroom-ready
        teaching package in seconds.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# Agent Overview
# ==============================

st.markdown("### 🤖 How EduAgent AI works")

agent_cols = st.columns(4)

agents = [
    ("📋", "Planning Agent", "Builds the lesson structure"),
    ("🧠", "Content Agent", "Creates teaching material"),
    ("📝", "Assessment Agent", "Creates questions & homework"),
    ("🔍", "Review Agent", "Checks quality and alignment"),
]

for col, (icon, name, description) in zip(agent_cols, agents):
    with col:
        st.markdown(
            f"""
            <div class="agent-card">
                <div class="agent-icon">{icon}</div>
                <div class="agent-name">{name}</div>
                <div class="agent-desc">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("### 🧑‍🏫 Create your teaching package")
st.markdown(
    '<div class="section-note">Enter your lesson details below, then let the four AI agents prepare the package.</div>',
    unsafe_allow_html=True
)


with st.form("teaching_package_form"):

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        subject = st.text_input(
            "Subject",
            value="Physics",
            key="subject_input",
            placeholder="e.g. Physics"
        )

        grade = st.text_input(
            "Grade",
            value="Grade 9",
            key="grade_input",
            placeholder="e.g. Grade 9"
        )

        topic = st.text_input(
            "Topic",
            value="Ohm's Law",
            key="topic_input",
            placeholder="e.g. Ohm's Law"
        )

    with input_col2:
        duration = st.number_input(
            "Class Duration (minutes)",
            min_value=10,
            max_value=180,
            value=40,
            step=5,
            key="duration_input"
        )

        difficulty = st.selectbox(
            "Difficulty Level",
            ["Easy", "Medium", "Hard"],
            index=1,
            key="difficulty_input"
        )

        st.caption("💡 Tip: Choose a difficulty level appropriate for your students.")

    st.markdown("### 📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF (curriculum, textbook, teacher notes, etc.)",
        type=["pdf"]
    )

    submitted = st.form_submit_button(
        "🚀 Generate Teaching Package",
        type="primary",
        use_container_width=True
    )


# ==============================
# Generate Package
# ==============================

if submitted:

    knowledge_text = ""

    if uploaded_file is not None:
        with st.spinner("📚 Reading your PDF..."):
            knowledge_text = extract_pdf_text(uploaded_file)

        if not knowledge_text.strip():
            st.error("❌ Could not extract readable text from this PDF.")
            st.stop()

        st.success(f"✅ PDF loaded successfully: {uploaded_file.name}")

    with st.spinner(
        "🤖 Four AI agents are preparing your teaching package..."
    ):

        package = generate_teaching_package(
            subject,
            grade,
            topic,
            duration,
            difficulty
        )

    st.success("✅ Teaching package generated successfully!")

    st.markdown(f"### 📚 Teaching Package: {topic}")

    st.caption(
        f"{subject} • {grade} • {duration} minutes • {difficulty} difficulty"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Lesson Plan",
        "🧠 Teaching Content",
        "📝 Assessment",
        "🔍 Review"
    ])

    with tab1:
        st.markdown(package["lesson_plan"])

    with tab2:
        st.markdown(package["content"])

    with tab3:
        st.markdown(package["assessment"])

    with tab4:
        st.markdown(package["review"])

    download_text = f"""EDUAGENT AI — TEACHING PACKAGE

Subject: {subject}
Grade: {grade}
Topic: {topic}
Class Duration: {duration} minutes
Difficulty: {difficulty}

==================================================
LESSON PLAN
==================================================

{package["lesson_plan"]}

==================================================
TEACHING CONTENT
==================================================

{package["content"]}

==================================================
ASSESSMENT
==================================================

{package["assessment"]}

==================================================
REVIEW
==================================================

{package["review"]}
"""

    st.download_button(
        "📥 Download Teaching Package",
        data=download_text,
        file_name=f"EduAgent_{topic.replace(' ', '_')}.txt",
        mime="text/plain",
        type="secondary",
        use_container_width=True
    )