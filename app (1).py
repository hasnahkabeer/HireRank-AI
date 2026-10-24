
import re
import io
import fitz
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. APP CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="HireRank AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 HireRank AI")
st.caption(
    "AI-assisted resume matching with transparent, "
    "job-related evidence."
)

SKILLS = [
    "python", "sql", "statistics", "data cleaning",
    "data visualization", "exploratory data analysis",
    "power bi", "tableau", "machine learning", "pandas",
    "numpy", "excel", "deep learning", "scikit-learn",
    "tensorflow", "pytorch", "nlp", "data analysis",
    "data engineering", "communication", "aws", "azure", "git"
]


# -------------------------------
# 2. LOAD SEMANTIC MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------------------
# 3. SKILL EXTRACTION
# -------------------------------

def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found.append(skill)

    return found


def find_skill_evidence(text, skills):
    """Return a short text excerpt for each detected skill."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    evidence = {}

    for skill in skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        for line in lines:
            if re.search(pattern, line, flags=re.IGNORECASE):
                evidence[skill] = line[:240]
                break

    return evidence


# -------------------------------
# 4. PDF TEXT EXTRACTION
# -------------------------------

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


# -------------------------------
# 5. EXPLAINABLE RANKING
# -------------------------------

def rank_candidates(job_description, candidates):
    model = load_model()

    job_skills = set(extract_skills(job_description))

    if not job_skills:
        st.warning(
            "No recognized skills were detected in the job "
            "description. Add skills such as Python or SQL."
        )
        return pd.DataFrame(), {}

    job_vector = model.encode(
        [job_description],
        normalize_embeddings=True
    )

    resume_texts = [c["text"] for c in candidates]

    resume_vectors = model.encode(
        resume_texts,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        resume_vectors, job_vector
    ).flatten()

    results = []
    explanations = {}

    for candidate, similarity in zip(candidates, similarities):
        text = candidate["text"]
        candidate_skills = set(extract_skills(text))

        matched = sorted(job_skills & candidate_skills)
        missing = sorted(job_skills - candidate_skills)

        skill_score = len(matched) / len(job_skills)

        # Cosine similarity of normalized sentence embeddings.
        # Clipping avoids negative similarity scores.
        semantic_score = max(0.0, min(1.0, float(similarity)))

        final_score = (
            0.50 * skill_score +
            0.50 * semantic_score
        ) * 100

        evidence = find_skill_evidence(text, matched)

        results.append({
            "Candidate": candidate["name"],
            "Match Score (%)": round(final_score, 2),
            "Skill Match (%)": round(skill_score * 100, 2),
            "Semantic Similarity (%)": round(
                semantic_score * 100, 2
            ),
            "Matched Skills": ", ".join(matched) or "None detected",
            "Missing Skills": ", ".join(missing) or "None detected"
        })

        explanations[candidate["name"]] = {
            "matched": matched,
            "missing": missing,
            "evidence": evidence,
            "skill_score": skill_score,
            "semantic_score": semantic_score,
            "text": text
        }

    results_df = pd.DataFrame(results).sort_values(
        "Match Score (%)",
        ascending=False
    ).reset_index(drop=True)

    return results_df, explanations


# -------------------------------
# 6. SIDEBAR INPUTS
# -------------------------------

st.sidebar.header("Job requirements")

job_description = st.sidebar.text_area(
    "Paste the job description",
    height=260,
    placeholder=(
        "Example: We need a Data Analyst with Python, "
        "SQL, statistics, and data visualization skills."
    )
)

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.info(
    "This prototype supports text-based PDFs. "
    "Scanned PDFs need OCR."
)


# -------------------------------
# 7. PROCESS RESUMES
# -------------------------------

if st.button("Analyze and rank resumes", type="primary"):

    if not job_description.strip():
        st.error("Please enter a job description.")

    elif not uploaded_pdfs:
        st.error("Please upload at least one PDF resume.")

    else:
        candidates = []
        seen_names = set()

        for pdf in uploaded_pdfs:
            if pdf.name in seen_names:
                st.warning(
                    f"Skipped duplicate filename: {pdf.name}"
                )
                continue

            seen_names.add(pdf.name)
            text = read_pdf(pdf)

            if text:
                candidates.append({
                    "name": pdf.name,
                    "text": text
                })
            else:
                st.warning(
                    f"No readable text found in {pdf.name}."
                )

        if not candidates:
            st.error(
                "No readable resumes were found. "
                "Try text-based PDFs."
            )

        else:
            with st.spinner("Analyzing resumes with AI..."):
                results_df, explanations = rank_candidates(
                    job_description,
                    candidates
                )

            if not results_df.empty:
                st.session_state["results"] = results_df
                st.session_state["explanations"] = explanations
                st.session_state["job_description"] = job_description


# -------------------------------
# 8. RESULTS DASHBOARD
# -------------------------------

if "results" in st.session_state:

    results_df = st.session_state["results"]
    explanations = st.session_state["explanations"]

    st.subheader("Ranking overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Resumes analyzed", len(results_df))
    col2.metric(
        "Highest match score",
        f"{results_df['Match Score (%)'].max():.2f}%"
    )
    col3.metric(
        "Average match score",
        f"{results_df['Match Score (%)'].mean():.2f}%"
    )

    st.subheader("Candidate ranking")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download ranking as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="hirerank_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Scores represent experimental text and skill matching, "
        "not candidate quality or hiring probability."
    )

    # ---------------------------
    # 9. CANDIDATE COMPARISON
    # ---------------------------

    st.divider()
    st.subheader("Compare candidates")

    names = results_df["Candidate"].tolist()

    selected = st.multiselect(
        "Choose two candidates",
        options=names,
        default=names[:2],
        max_selections=2
    )

    if len(selected) == 2:

        left, right = st.columns(2)

        for column, name in zip([left, right], selected):
            info = explanations[name]

            with column:
                row = results_df[
                    results_df["Candidate"] == name
                ].iloc[0]

                st.markdown(f"### {name}")
                st.metric(
                    "Match score",
                    f"{row['Match Score (%)']:.2f}%"
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
                    st.write("All recognized job skills detected.")

                st.write("**Why skills were matched**")

                if info["evidence"]:
                    for skill, snippet in info["evidence"].items():
                        st.markdown(f"- **{skill}:** {snippet}")
                else:
                    st.write("No skill evidence was extracted.")

                with st.expander("Score breakdown"):
                    st.write(
                        f"Skill overlap: "
                        f"{info['skill_score']:.1%}"
                    )
                    st.write(
                        f"Semantic similarity: "
                        f"{info['semantic_score']:.1%}"
                    )

    elif len(selected) == 1:
        st.info("Choose one more candidate to compare.")

    st.divider()
    st.subheader("How to interpret the results")

    st.markdown("""
    - **Skill overlap:** fraction of recognized job skills found
      in the resume.
    - **Semantic similarity:** similarity between resume text
      and the job description using sentence embeddings.
    - **Matched skills:** terms found in both documents.
    - **Missing skills:** job-description skills not detected
      in the extracted resume text.
    - **Evidence:** a short excerpt showing where a matched
      skill was detected.

    Missing skills do not prove that a candidate lacks them.
    PDF extraction and the skill dictionary can miss information.
    Review original resumes before making any decisions.
    """)
