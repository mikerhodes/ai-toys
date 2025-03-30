"""
copyeditor.py

This toy allows AI copyediting of files. The thing that makes it
ergonomic in my workflow is that it automatically loads the
last edited file at startup and uses that as the copy to evaluate,
avoiding the need to manually find the file --- because it is,
without fail, going to be the one you edited last because presumably
you are editing it concurrently with the copy edit session.
"""

import logging
import os

from typing import Optional

import anthropic
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH = "/Users/mike/code/gh/mikerhodes/dx13-hugo/content"

st.set_page_config(layout="wide")

PROMPT = """
You are an AI copyeditor with a keen eye for detail and a deep understanding of language, style, and grammar. Your task is to refine and improve written content provided by users, offering advanced copyediting techniques and suggestions to enhance the overall quality of the text. When a user submits a piece of writing, follow these steps:

1. Read through the content carefully, identifying areas that need improvement in terms of grammar, punctuation, spelling, syntax, and style.

2. Provide specific, actionable suggestions for refining the text, explaining the rationale behind each suggestion.

3. Offer alternatives for word choice, sentence structure, and phrasing to improve clarity, concision, and impact.

4. Ensure the tone and voice of the writing are consistent and appropriate for the intended audience of senior programmers.

5. Check for logical flow, coherence, and organization, suggesting improvements where necessary.

6. Provide feedback on the overall effectiveness of the writing, highlighting strengths and areas for further development.

7. Highlight changes you have made using this format: `:red-background[my changes here]`

Your suggestions should be constructive, insightful, and designed to help the user elevate the quality of their writing.

Avoid use of headings in your response.

The date today is 2025-03-26.
"""


def find_most_recent_file(directory) -> Optional[str]:
    """Find the most recently modified file in a directory tree."""
    most_recent_file = None
    most_recent_time = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(".md"):
                continue
            try:
                # Get the most recent of modification or creation time
                mtime = os.path.getmtime(file_path)
                ctime = os.path.getctime(file_path)
                file_time = max(mtime, ctime)

                if file_time > most_recent_time:
                    most_recent_time = file_time
                    most_recent_file = file_path
            except (OSError, PermissionError):
                # Handle cases where we can't access the file
                continue

    if most_recent_file:
        return most_recent_file
    return None


def load_markdown_without_frontmatter(file_path: str):
    content = []
    in_frontmatter = False

    with open(file_path, "r", encoding="utf-8") as file:
        first_line = file.readline()

        # Check if file starts with frontmatter
        if first_line.strip() == "---":
            in_frontmatter = True
        else:
            # No frontmatter, include the first line
            content.append(first_line)

        # Process the rest of the file
        for line in file:
            if in_frontmatter and line.strip() == "---":
                in_frontmatter = False
                continue
            if not in_frontmatter:
                content.append(line)

    return "".join(content)


if "markdown_content" not in st.session_state:
    st.session_state["markdown_content"] = ""
if "copyedited_content" not in st.session_state:
    st.session_state["copyedited_content"] = ""

#
# Streamlit UI
#


def copyedit_with_claude():
    # Display a spinner while waiting for Claude
    with st.spinner("Claude is copyediting your markdown..."):
        try:
            # Initialize the Claude client
            # You should store your API key in st.secrets or as an environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                st.error(
                    "No API key found. Please set the ANTHROPIC_API_KEY in Streamlit secrets or environment variables."
                )
                logger.error("Set ANTHROPIC_API_KEY")
                return

            client = anthropic.Anthropic(api_key=api_key)

            # Call Claude with the appropriate prompt
            response = client.messages.create(
                model="claude-3-7-sonnet-latest",  # or another Claude model
                max_tokens=4000,
                temperature=0.0,  # Keep it deterministic for copyediting
                system=PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please copyedit this markdown text:\n\n<content>{st.session_state.markdown_content}</content>",
                    }
                ],
            )

            # Store the copyedited content
            st.session_state.copyedited_content = response.content[0].text

        except Exception as e:
            st.error(f"Error during copyediting: {str(e)}")


# Add tabs for original and copyedited content
tab1, tab2 = st.columns(2)

with tab1:
    f = find_most_recent_file(PATH)

    if f:
        st.info(
            f"Loaded file `{f.lstrip(PATH)}`.", icon=":material/article:"
        )

    foo = st.empty()

    # Display the original markdown content
    @st.fragment(run_every=1)
    def display_markdown():
        with foo:
            if f:
                st.session_state.markdown_content = (
                    load_markdown_without_frontmatter(f)
                )
                st.container(border=True).markdown(
                    st.session_state.markdown_content
                )

    # if "running" not in st.session_state:
    display_markdown()

with tab2:
    if not st.session_state.copyedited_content:
        st.info(
            "Click the 'Copyedit with Claude' button to see the copyedited version here.",
            icon=":material/robot_2:",
        )
    if st.button("Copyedit with Claude", use_container_width=True):
        copyedit_with_claude()
    if st.session_state.copyedited_content:
        st.container(border=True).markdown(
            st.session_state.copyedited_content
        )
