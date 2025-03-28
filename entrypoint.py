import logging
import os
import time

from typing import Optional

import anthropic
import streamlit as st
from watchdog.observers import Observer
from watchdog.events import (
    FileCreatedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH = "/Users/mike/code/gh/mikerhodes/dx13-hugo/content"

st.set_page_config(layout="wide")


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


class MarkdownHandler(FileSystemEventHandler):
    def __init__(self):
        self.markdown_content = ""
        self.last_updated_file = ""
        self.last_update_time = None

    def on_modified(self, event):
        logger.info(event)
        match event:
            case FileModifiedEvent():
                p = str(event.src_path)
                self._on_event(p)

    def on_created(self, event):
        logger.info(event)
        match event:
            case FileCreatedEvent():
                p = str(event.src_path)
                self._on_event(p)

    def _on_event(self, p: str):
        if p.endswith(".md"):
            try:
                self.markdown_content = load_markdown_without_frontmatter(p)
                self.last_updated_file = os.path.basename(p)
                self.last_update_time = time.strftime("%H:%M:%S")
            except Exception as e:
                print(f"Error reading file: {e}")


logger.info("hello")

# Initialize the file watcher in session state
# if "watcher_initialized" not in st.session_state:
#     logger.info("Starting watcher on %s", PATH)
#     st.session_state.handler = MarkdownHandler()
#     st.session_state.watcher_initialized = True

#     # Directory to monitor
#     path_to_watch = PATH

#     # Create the directory if it doesn't exist
#     os.makedirs(path_to_watch, exist_ok=True)

#     # Set up the observer in a background thread
#     observer = Observer()
#     observer.schedule(
#         st.session_state.handler, path_to_watch, recursive=True
#     )
#     observer.daemon = (
#         True  # Set as daemon so it stops when the main thread stops
#     )
#     observer.start()
#     logger.info("Started watcher on %s", PATH)

if "copyedited_content" not in st.session_state:
    st.session_state["copyedited_content"] = ""

# Streamlit UI
st.title("Claude Copyeditor")


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
                system="""
You are an AI copyeditor with a keen eye for detail and a deep understanding of language, style, and grammar. Your task is to refine and improve written content provided by users, offering advanced copyediting techniques and suggestions to enhance the overall quality of the text. When a user submits a piece of writing, follow these steps:

1. Ignore the markdown frontmatter, if there is any.

2. Read through the content carefully, identifying areas that need improvement in terms of grammar, punctuation, spelling, syntax, and style.

3. Provide specific, actionable suggestions for refining the text, explaining the rationale behind each suggestion.

4. Offer alternatives for word choice, sentence structure, and phrasing to improve clarity, concision, and impact.

5. Ensure the tone and voice of the writing are consistent and appropriate for the intended audience of senior programmers.

6. Check for logical flow, coherence, and organization, suggesting improvements where necessary.

7. Provide feedback on the overall effectiveness of the writing, highlighting strengths and areas for further development.

Your suggestions should be constructive, insightful, and designed to help the user elevate the quality of their writing.

The date today is 2025-03-26.
                """,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please copyedit this markdown text:\n\n{st.session_state.handler.markdown_content}",
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
    # Display file info if content exists
    if st.session_state.handler.markdown_content:
        st.success(
            f"Last updated: {st.session_state.handler.last_updated_file} at {st.session_state.handler.last_update_time}"
        )
    foo = st.empty()

    if f := find_most_recent_file(PATH):
        logger.info(f)
        st.markdown(load_markdown_without_frontmatter(f))

    # Display the original markdown content
    # @st.fragment(run_every=1)
    # def display_markdown():
    #     # st.session_state.running = True
    #     # logger.info("dm")
    #     with foo:
    #         if st.session_state.handler.markdown_content:
    #             st.markdown(st.session_state.handler.markdown_content)
    #         else:
    #             st.info("Waiting for markdown files to be modified...")

    # if "running" not in st.session_state:
    # display_markdown()
    # if st.session_state.handler.markdown_content:
    #     st.markdown(st.session_state.handler.markdown_content)
    # else:
    #     st.info("Waiting for markdown files to be modified...")

with tab2:
    # Add a button to copyedit the content
    if st.button("Copyedit with Claude", use_container_width=True):
        copyedit_with_claude()

    if st.session_state.copyedited_content:
        st.markdown(st.session_state.copyedited_content)
    else:
        st.info(
            "Click the 'Copyedit with Claude' button to see the copyedited version here."
        )


# Display monitoring status
st.subheader("Monitor Status")
st.write(f"Monitoring directory: {os.path.abspath(PATH)}")
