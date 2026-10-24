# HireRank AI — Explainable Resume Matching

HireRank AI is a Streamlit-based application that compares resumes against a job description using skill matching and semantic text similarity.

## Features

- Upload multiple PDF resumes.
- Extract text from resumes.
- Detect job-related skills.
- Calculate experimental resume relevance scores.
- Compare candidates side by side.
- Display matched skills, skills not detected, and resume evidence.
- Export ranking results to CSV.

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- pandas
- scikit-learn
- Sentence Transformers

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Limitations and Responsible Use

The scores are experimental text-matching indicators, not validated hiring probabilities. Skill detection can miss relevant information, and semantic similarity does not establish candidate competence. The application is intended to support human review, not automatic rejection or hiring decisions.

Use synthetic resumes for testing and avoid uploading personal candidate data to public repositories.
