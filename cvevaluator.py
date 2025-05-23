"""
evevaluator.py

This ai-toy allows evaluating a CV against a job description.
"""

import logging
import os

import anthropic
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVALUATE_HELPER_PROMPT = """
Evaluate this CV against the job description.

- Is it worth applying for this role?
- Where are there strong matches?
- Where are there partial matches?
- What are potential gaps?
- What recommendations would you have for the candidate applying?
    - What parts of their experience should they emphasise?

<cv>
{cv}
</cv>

<jobdescription>
{jobdesc}
</jobdescription>
"""
EVALUATE_PROMPT = """
Imagine you are a senior hiring manager for a tech firm. You are rushed off your feet and don't have much time to read the CV. In a quick scan, does it meet the job description?

Evaluate the CV from this point of view, whether it would stand out as worth taking forward from a large set of applicants. Provide feedback on the CV from this point of view.

<cv>
{cv}
</cv>

<jobdescription>
{jobdesc}
</jobdescription>
"""
IMPROVE_PROMPT = """
Imagine you are a senior recruitment partner. There are a lot of applicants for this role, and you want to help your candidate stand out from the field.

Provide recommendations to this candidate for how to improve their CV to match the given role. Think about:

- Writing style.
- Clarity of the CV --- is it easy to understand the candidates strengths, or are they hidden in a pile of words?
- Alternative ways of presenting experience to better align with the job needs.
- Additional experience the candidate may want to consider obtaining.

<cv>
{cv}
</cv>

<jobdescription>
{jobdesc}
</jobdescription>
"""
COVER_PROMPT = """
Imagine you are a senior recruitment partner.

Help the candidate focus their cover letter on the needs in the job description based on the experience in their CV.

A typical cover letter will discuss:

- Why the candidate wants this job in particular.
- Why their skills would be valuable to the company.
- How their unique skill set makes them a strong candidate for this position.

<cv>
{cv}
</cv>

<jobdescription>
{jobdesc}
</jobdescription>
"""

st.set_page_config(layout="wide")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("upload", type=["txt", "md"])
    if uploaded_file is not None:
        st.session_state.cv = uploaded_file.getvalue().decode("utf-8")
    st.text_area("CV (markdown is best)", key="cv", height=400)

with col2:
    st.text_area(
        "Job description (markdown is best)", key="jobdesc", height=400
    )

st.session_state.do_task = None


do_task = False
do_prompt = None

cols = st.columns(4)

with cols[0]:
    if st.button("Evaluate CV against Job Description as a Helper"):
        do_task = True
        do_prompt = EVALUATE_HELPER_PROMPT
with cols[1]:
    if st.button("Evaluate CV against Job Description as a Hiring Manager"):
        do_task = True
        do_prompt = EVALUATE_PROMPT
with cols[2]:
    if st.button("Provide CV recommendations for job description"):
        do_prompt = IMPROVE_PROMPT
        do_task = True
with cols[3]:
    if st.button("Provide cover letter help for this job description"):
        do_prompt = COVER_PROMPT
        do_task = True

if do_task and do_prompt:
    with st.spinner("Claude is carrying out your task..."):
        try:
            # Initialize the Claude client
            # You should store your API key in st.secrets or as an environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                st.error(
                    "No API key found. Please set the ANTHROPIC_API_KEY in Streamlit secrets or environment variables."
                )
                logger.error("Set ANTHROPIC_API_KEY")
                st.stop()

            client = anthropic.Anthropic(api_key=api_key)

            # Call Claude with the appropriate prompt
            response = client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=4000,
                temperature=0.5,
                messages=[
                    {
                        "role": "user",
                        "content": do_prompt.format(
                            cv=st.session_state.cv,
                            jobdesc=st.session_state.jobdesc,
                        ),
                    }
                ],
            )

            # Store the copyedited content
            st.session_state.evaluation = response.content[0].text

        except Exception as e:
            st.error(f"Error during copyediting: {str(e)}")

if "evaluation" in st.session_state:
    _, c, _ = st.columns([1, 3, 1])
    with c:
        st.write(st.session_state.evaluation)
