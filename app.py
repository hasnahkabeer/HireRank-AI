
import re
import fitz
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="HireRank AI | Resume Intelligence",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 HireRank AI")
st.caption(
    "Explainable AI-assisted resume matching "
    "| Human review required"
)

SKILLS = [
    "python", "sql", "statistics", "data cleaning",
    "data visualization", "exploratory data analysis",
    "power bi", "tableau", "machine learning", "pandas",
    "numpy", "excel", "deep learning", "scikit-learn",
    "tensorflow", "pytorch", "nlp", "data analysis",
    "data engineering", "communication", "aws", "azure", "git"
]


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# SKILL EXTRACTION
# -----------------------------
def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def extract_evidence(text, skills):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    evidence = {}

    for skill in skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        for line in lines:
            if re.search(pattern, line, re.IGNORECASE):
                evidence[skill] = line[:250]
                break

    return evidence


# -----------------------------
# PDF READER
# -----------------------------
def read_pdf(uploaded_file):
    try:
        with fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        ) as pdf:
            return "\n".join(
                page.get_text() for page in pdf
            ).strip()
    except Exception:
        return ""


# -----------------------------
# RANKING ENGINE
# -----------------------------
def rank_candidates(job_description, candidates):
    model = load_model()

    job_skills = set(extract_skills(job_description))

    if not job_skills:
        raise ValueError(
            "No recognized skills were found in the job description. "
            "Add skills such as Python, SQL or Power BI."
        )

    job_vector = model.encode(
        [job_description],
        normalize_embeddings=True
    )

    resume_vectors = model.encode(
        [c["text"] for c in candidates],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        resume_vectors, job_vector
    ).flatten()

    rows = []
    explanations = {}

    for candidate, similarity in zip(candidates, similarities):
        candidate_skills = set(
            extract_skills(candidate["text"])
        )

        matched = sorted(job_skills & candidate_skills)
        missing = sorted(job_skills - candidate_skills)

        skill_score = len(matched) / len(job_skills)

        # Cosine similarity can be negative or positive.
        # Keep the raw score visible; do not call it a probability.
        semantic_score = float(similarity)

        # Initial experimental weights, not validated hiring criteria.
        # Shift cosine similarity from [-1, 1] to [0, 1].
        semantic_component = max(
            0.0, min(1.0, (semantic_score + 1) / 2)
        )

        final_score = (
            0.50 * skill_score +
            0.50 * semantic_component
        ) * 100

        evidence = extract_evidence(
            candidate["text"], matched
        )

        rows.append({
            "Candidate": candidate["name"],
            "Match Score (%)": round(final_score, 1),
            "Skill Match (%)": round(skill_score * 100, 1),
            "Semantic Similarity": round(semantic_score, 3),
            "Matched Skills": ", ".join(matched) or "None detected",
            "Skills Not Detected": ", ".join(missing) or "None detected"
        })

        explanations[candidate["name"]] = {
            "matched": matched,
            "missing": missing,
            "evidence": evidence,
            "skill_score": skill_score,
            "semantic_score": semantic_score,
            "text": candidate["text"]
        }

    df = pd.DataFrame(rows).sort_values(
        "Match Score (%)", ascending=False
    ).reset_index(drop=True)

    return df, explanations


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Job requirements")

job_description = st.sidebar.text_area(
    "Job description",
    height=260,
    placeholder=(
        "Paste the job description here, including "
        "required skills and responsibilities."
    )
)

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.info(
    "Use synthetic resumes for testing. "
    "Scanned PDFs require OCR."
)


# -----------------------------
# ANALYZE
# -----------------------------
if st.sidebar.button("Analyze resumes", type="primary"):

    if not job_description.strip():
        st.sidebar.error("Enter a job description first.")

    elif not uploaded_pdfs:
        st.sidebar.error("Upload at least one PDF resume.")

    else:
        candidates = []
        seen = set()

        for pdf in uploaded_pdfs:
            if pdf.name in seen:
                continue

            seen.add(pdf.name)
            text = read_pdf(pdf)

            if text:
                candidates.append({
                    "name": pdf.name,
                    "text": text
                })
            else:
                st.warning(
                    f"Could not extract text from {pdf.name}."
                )

        if not candidates:
            st.error("No readable PDF resumes were found.")

        else:
            try:
                with st.spinner(
                    "Loading AI model and analyzing resumes..."
                ):
                    df, explanations = rank_candidates(
                        job_description, candidates
                    )

                st.session_state["results"] = df
                st.session_state["explanations"] = explanations

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# -----------------------------
# DASHBOARD
# -----------------------------
if "results" in st.session_state:

    df = st.session_state["results"]
    explanations = st.session_state["explanations"]

    st.divider()
    st.header("📊 Ranking overview")

    a, b, c = st.columns(3)

    a.metric("Resumes analyzed", len(df))
    b.metric(
        "Highest match score",
        f"{df['Match Score (%)'].max():.1f}%"
    )
    c.metric(
        "Average match score",
        f"{df['Match Score (%)'].mean():.1f}%"
    )

    # -------------------------
    # RANKING CHART
    # -------------------------
    st.subheader("Candidate match comparison")

    chart_data = df.set_index("Candidate")[
        ["Match Score (%)"]
    ]

    st.bar_chart(chart_data, horizontal=True)

    # -------------------------
    # RESULTS TABLE
    # -------------------------
    st.subheader("Ranked candidates")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match Score (%)": st.column_config.ProgressColumn(
                "Match Score",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "Skill Match (%)": st.column_config.NumberColumn(
                "Skill Match",
                format="%.1f%%"
            )
        }
    )

    st.download_button(
        "⬇️ Download ranking CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hirerank_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Scores are experimental text-matching indicators, "
        "not probabilities of job success or candidate quality."
    )

    # -------------------------
    # CANDIDATE COMPARISON
    # -------------------------
    st.divider()
    st.header("🔍 Explainable candidate comparison")

    names = df["Candidate"].tolist()

    first_col, second_col = st.columns(2)

    with first_col:
        first = st.selectbox(
            "Candidate A",
            names,
            index=0,
            key="candidate_a"
        )

    with second_col:
        default_second = 1 if len(names) > 1 else 0

        second = st.selectbox(
            "Candidate B",
            names,
            index=default_second,
            key="candidate_b"
        )

    if first == second:
        st.info("Select two different candidates to compare.")

    else:
        left, right = st.columns(2)

        for column, name in [(left, first), (right, second)]:
            info = explanations[name]

            row = df[df["Candidate"] == name].iloc[0]

            with column:
                st.subheader(name)

                st.metric(
                    "Experimental match score",
                    f"{row['Match Score (%)']:.1f}%"
                )

                st.write("**Matched skills**")

                if info["matched"]:
                    st.write(", ".join(info["matched"]))
                else:
                    st.write("No recognized skills detected.")

                st.write("**Skills not detected**")

                if info["missing"]:
                    st.write(", ".join(info["missing"]))
                else:
                    st.write("All recognized job skills were detected.")

                st.write("**Evidence from resume**")

                if info["evidence"]:
                    for skill, snippet in info["evidence"].items():
                        st.markdown(f"**{skill}**")
                        st.caption(snippet)
                else:
                    st.write("No skill evidence extracted.")

                with st.expander("View score breakdown"):
                    st.write(
                        f"Skill overlap: {info['skill_score']:.1%}"
                    )
                    st.write(
                        "Raw cosine similarity: "
                        f"{info['semantic_score']:.3f}"
                    )

                    st.caption(
                        "The final score combines skill overlap "
                        "and a shifted semantic similarity score."
                    )

    # -------------------------
    # REVIEW GUIDANCE
    # -------------------------
    st.divider()
    st.subheader("Responsible use")

    st.markdown("""
    - Review the original resume before drawing conclusions.
    - "Skills not detected" means the system did not find a
      recognized skill term; it does not prove the candidate
      lacks that skill.
    - The skill dictionary may miss synonyms and new technologies.
    - Scores and weights have not been validated against hiring
      outcomes. Do not use them for automatic rejection.
    - Avoid using protected characteristics or personal details
      as ranking criteria.
    """)

else:
    st.info(
        "Enter a job description, upload resumes, and click "
        "**Analyze resumes** in the sidebar to begin."
    )

