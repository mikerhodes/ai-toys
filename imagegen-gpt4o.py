# imagegen-gpt4o is a bit of a frankenstein's code thing.
#
# The Streamlit code is a bit messy too, but I've cleaned it
# up and reworked the UX a whole lot from Claude's original
# suggested code, which somewhat arbitrarily put some options
# into a sidebar.
#
# Overall, it's another script that Claude's start on was okay,
# and really was useful in getting started, but required a
# bunch of work to make it nicer to use (and update).
import base64
import logging
import os
import textwrap
from typing import Optional, cast

import openai
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="AI Image Generator", page_icon="🎨", layout="wide"
)

# Get API key from environment variable, with option to override in UI
api_key = os.getenv("OPENAI_API_KEY", "")


# Session state initialization
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = None


class State:
    generated_image: bytes | None
    error_message: str | None
    last_user_prompt: str | None


# Strongly typed session state
_s = cast(State, st.session_state)


# Function to generate an image with DALL-E 3
def generate_image(
    prompt,
    api_key,
    size=None,
    quality=None,
    background="auto",
    uploaded_images=None,
) -> Optional[bytes]:
    client = openai.OpenAI(api_key=api_key)

    import time

    logger.info("Generating image...")
    start_time = time.time()

    try:
        if uploaded_images:
            logger.info("Editing image...")
            response = client.images.edit(
                model="gpt-image-1",
                image=uploaded_images,
                prompt=prompt,
                size=size,
                quality=quality,  # type: ignore
                background=background,  # type: ignore
                n=1,
            )
        else:
            logger.info("Creating a new image...")
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,  # type: ignore
                output_format="png",
                background=background,  # type: ignore
                n=1,
                moderation="low",
            )
        time_taken = time.time() - start_time
        logger.info(f"OpenAI image call took {time_taken:.4f} seconds")
        if not response.data:
            return None

        image_base64 = response.data[0].b64_json
        if image_base64 is None:
            return None

        image_bytes = base64.b64decode(image_base64)

        return image_bytes
    except Exception as e:
        logger.error("Error generating image: %s", e)
        return None


def process_image_generation(prompt, uploaded_images):
    with st.spinner("Generating image...", show_time=True):
        try:
            # Generate the image with DALL-E 3
            result = generate_image(
                prompt,
                api_key,
                size=image_size,
                quality=image_quality,
                background=image_background,
                uploaded_images=uploaded_images,
            )

            if result is None:
                st.session_state.error_message = "Error generating image"
                _s.generated_image = None
            else:
                _s.generated_image = result
                st.session_state.error_message = None
                st.session_state.last_user_prompt = prompt

        except Exception as e:
            st.session_state.error_message = str(e)
            _s.generated_image = None


#
# Main layout
#

if not api_key:
    st.warning(
        "Please set the OPENAI_API_KEY environment variable",
        icon="⚠️",
    )

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    uploaded_files = st.file_uploader(
        "(Optional) Upload image(s) to edit",
        accept_multiple_files=True,
        type=["png", "webp", "jpg"],
    )
    user_prompt = st.text_area(
        "Image description",
        height=200,
        value=textwrap.dedent("""
        Generate an anime-style image of a gigantic cat striding through
        a lush forest, brightly lit with sunshine. The cat looks really
        cute even though it is huge. 
        """).strip(),
    )

    def labels(x):
        return {
            "auto": "Auto",
            "1024x1024": "Square",
            "1024x1536": "Portrait",
            "1536x1024": "Landscape",
        }[x]

    image_size = st.radio(
        "Aspect ratio",
        ["auto", "1024x1024", "1024x1536", "1536x1024"],
        key="visibility",
        horizontal=True,
        format_func=labels,
    )

    # Add two buttons side by side for generating and regenerating
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        generate_button = st.button(
            "Generate Image",
            type="primary",
            use_container_width=True,
            disabled=not api_key,
        )
    with button_col2:
        regenerate_button = st.button(
            "Regenerate Image",
            use_container_width=True,
            disabled=not api_key,
        )

    # Options for advanced users
    with st.expander("Advanced Options"):
        image_quality = st.radio(
            "Image Quality",
            ["auto", "high", "meduim", "low"],
            index=0,
            horizontal=True,
        )
        image_background = st.radio(
            "Image background",
            ["auto", "opaque", "transparent"],
            index=0,
            horizontal=True,
        )


# Process regeneration when regenerate button is clicked
if regenerate_button:
    if st.session_state.last_user_prompt:
        # Regenerate using the same prompt
        process_image_generation(
            st.session_state.last_user_prompt, uploaded_files
        )
    elif user_prompt:
        # If no previous prompt but user has entered a new one
        process_image_generation(user_prompt, uploaded_files)
    else:
        st.error(
            "No prompt available for regeneration. Please enter a prompt first."
        )

with col2:
    # Process generation when button is clicked
    if generate_button and user_prompt:
        process_image_generation(user_prompt, uploaded_files)

    # Display the generated image or error message
    if st.session_state.error_message:
        st.error(f"Error: {st.session_state.error_message}")

    if _s.generated_image:
        # Display the image
        st.image(_s.generated_image, use_container_width=True)
        # Download button
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                label="Download Image",
                data=_s.generated_image,
                file_name="generated_image.png",
                mime="image/png",
            )
    else:
        st.markdown(
            """
            Generated image will appear here.
            """
        )

# Footer
st.markdown("---")
st.caption("Built with Streamlit and OpenAI API")
