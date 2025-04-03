import streamlit as st

st.set_page_config(
    page_title="Resume Screening",
    page_icon="📄"
)

st.title("Resume screening")

st.subheader("Resume Analyzer Benefits and Limitations")

st.markdown("""
### Benefits

**Highlight strengths:** Helps you identify the strongest aspects of your resume, such as relevant skills and experience.

**Improve keyword targeting:** Ensures your resume includes keywords that Applicant Tracking Systems (ATS) used by many companies look for, increasing your chances of getting past the initial screening.

**Structure and clarity:** Provides suggestions for better structure and clarity in your resume, making it easier for recruiters to understand your qualifications.

**Identify weaknesses:** Points out areas for improvement, such as lack of quantifiable achievements or unclear action verbs.

**Tailoring:** Can help tailor your resume for specific job applications by highlighting relevant skills and experience.

### Limitations

**Accuracy:** The effectiveness of a resume analyzer depends on the quality of its underlying algorithms.

**Complexity:** Complex achievements or nuanced experience might not be fully captured by the analysis.

**Human touch:** Resumes are often reviewed by humans, so a well-written, engaging resume is still important.

**Specificity:** Analyzers might not be specific to your target industry or job role.

### Overall

A resume analyzer is a valuable tool to get feedback on your resume and improve its effectiveness in getting you interviews. However, it should be used in conjunction with your own judgment and human review to ensure the best possible presentation of your skills and experience.
""")