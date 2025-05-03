# imagegen-dalle3 is a bit of a frankenstein's code thing.
#
# I'm not sure the idea of getting GPT-4 to improve the prompt
# is a good one, as dall-e 3 already does something like that.
# Also Claude's idea to get the improvement back as a tool call
# to a non-existent is kind of neat, but also a bit odd. But
# it does ensure that the model is providing what we want.
#
# The Streamlit code is a bit messy too, but I've cleaned it
# up and reworked the UX a whole lot from Claude's original
# suggested code, which somewhat arbitrarily put some options
# into a sidebar.
#
# Overall, it's another script that Claude's start on was okay,
# and really was useful in getting started, but required a
# bunch of work to make it nicer to use (and update).
import streamlit as st
import openai
import os
import json
import httpx
from typing import cast

# Set page configuration
st.set_page_config(
    page_title="AI Image Generator", page_icon="🎨", layout="wide"
)

# Get API key from environment variable, with option to override in UI
api_key = os.getenv("OPENAI_API_KEY", "")


# Session state initialization
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "optimized_prompt" not in st.session_state:
    st.session_state.optimized_prompt = None
if "dall_e_revised_prompt" not in st.session_state:
    st.session_state.dall_e_revised_prompt = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "last_user_prompt" not in st.session_state:
    st.session_state.last_user_prompt = None


class State:
    generated_image: str | None
    optimized_prompt: str | None
    dall_e_revised_prompt: str | None
    error_message: str | None
    last_user_prompt: str | None


# Strongly typed session state
_s = cast(State, st.session_state)


# Function to call GPT-4 with tool-calling capabilities
def get_optimized_prompt(user_prompt, api_key):
    client = openai.OpenAI(api_key=api_key)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a detailed prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed, descriptive prompt for DALL-E 3 to generate the image. Make it detailed and specific.",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": """You are an expert image prompt engineer. 
        Your job is to take a user's basic image request and convert it into a detailed, 
        descriptive prompt that will produce high-quality results with DALL-E 3.
        
        Include specific details about:
        - Style (photorealistic, digital art, oil painting, etc.)
        - Composition (close-up, wide shot, etc.)
        - Lighting (soft, dramatic, golden hour, etc.)
        - Mood (serene, energetic, mysterious, etc.)
        - Colors (vibrant, muted, specific color schemes)
        
        Always use the generate_image function to return your optimized prompt.
        """,
        },
        {
            "role": "user",
            "content": f"Create an image based on this idea: {user_prompt}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,  # type: ignore
        tools=tools,  # type: ignore
        tool_choice={
            "type": "function",
            "function": {"name": "generate_image"},
        },
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        function_args = json.loads(tool_calls[0].function.arguments)
        return function_args.get("prompt")
    else:
        return (
            user_prompt  # Fallback to original prompt if tool calling fails
        )


# Function to generate an image with DALL-E 3
def generate_image(prompt, api_key, size=None, quality=None, style=None):
    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,  # type: ignore
            style=style,
            n=1,
        )

        return {
            "url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt,
        }
    except Exception as e:
        return {"error": str(e)}


def process_image_generation(prompt_to_use, is_regenerating=False):
    with st.spinner("Generating image..."):
        try:
            # Determine which prompt to use
            if (
                is_regenerating
                and keep_optimized_prompt
                and st.session_state.optimized_prompt
            ):
                # Use the existing optimized prompt when regenerating (if option selected)
                optimized_prompt = st.session_state.optimized_prompt
            elif use_gpt4_optimization:
                # Generate a new optimized prompt
                with st.spinner("Optimizing prompt with GPT-4..."):
                    optimized_prompt = get_optimized_prompt(
                        prompt_to_use, api_key
                    )
                st.session_state.optimized_prompt = optimized_prompt
            else:
                # Use the original user prompt
                optimized_prompt = prompt_to_use
                st.session_state.optimized_prompt = None

            # Generate the image with DALL-E 3
            result = generate_image(
                optimized_prompt,
                api_key,
                size=image_size,
                quality=image_quality,
                style=image_style,
            )

            if "error" in result:
                st.session_state.error_message = result["error"]
                _s.generated_image = None
                st.session_state.dall_e_revised_prompt = None
            else:
                _s.generated_image = result["url"]
                st.session_state.dall_e_revised_prompt = result[
                    "revised_prompt"
                ]
                st.session_state.error_message = None
                st.session_state.last_user_prompt = prompt_to_use

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
    user_prompt = st.text_area(
        "Image description",
        height=200,
        value="""Generate a high-fantasy cell-shaded avatar image of a Caucasian female human mage. They have a time-worn face. They are wearing a richly detailed, dark travelling cloak with a simple golden clasp at the neck. Below the cloak, they are wearing well-worn, close-fitting trousers and scuffed, high travelling boots. They have richly coloured wisps of smoke-like magic surrounding them.""",
    )

    def labels(x):
        return {
            "1024x1024": "Square",
            "1024x1792": "Portrait",
            "1792x1024": "Landscape",
        }[x]

    image_size = st.radio(
        "Aspect ratio",
        ["1024x1024", "1024x1792", "1792x1024"],
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
        use_gpt4_optimization = st.checkbox(
            "Use GPT-4 to optimize prompt", value=False
        )
        show_dall_e_revised_prompt = st.checkbox(
            "Show DALL-E revised prompt", value=True
        )
        keep_optimized_prompt = st.checkbox(
            "Keep optimized prompt when regenerating", value=True
        )
        image_quality = st.radio(
            "Image Quality", ["standard", "hd"], index=0, horizontal=True
        )
        image_style = st.radio(
            "Image Style", ["vivid", "natural"], index=0, horizontal=True
        )


# Process generation when button is clicked
if generate_button and user_prompt:
    process_image_generation(user_prompt, is_regenerating=False)

# Process regeneration when regenerate button is clicked
if regenerate_button:
    if st.session_state.last_user_prompt:
        # Regenerate using the same prompt
        process_image_generation(
            st.session_state.last_user_prompt, is_regenerating=True
        )
    elif user_prompt:
        # If no previous prompt but user has entered a new one
        process_image_generation(user_prompt, is_regenerating=False)
    else:
        st.error(
            "No prompt available for regeneration. Please enter a prompt first."
        )

with col2:
    # Display the generated image or error message
    if st.session_state.error_message:
        st.error(f"Error: {st.session_state.error_message}")

    if _s.generated_image:
        # Display prompt information if available

        # Display the image
        st.image(_s.generated_image, use_container_width=True)
        # Download button
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                label="Download Image",
                data=httpx.get(_s.generated_image).content,
                file_name="generated_image.png",
                mime="image/png",
            )

        prompt_cols = st.columns(2)
        with prompt_cols[0]:
            if st.session_state.optimized_prompt and use_gpt4_optimization:
                st.markdown("##### GPT-4 Optimized Prompt:")
                st.info(st.session_state.optimized_prompt)
        with prompt_cols[1]:
            if (
                st.session_state.dall_e_revised_prompt
                and show_dall_e_revised_prompt
            ):
                st.markdown("##### DALL-E 3 Revised Prompt:")
                st.info(st.session_state.dall_e_revised_prompt)

    else:
        st.markdown(
            """
            Generated image will appear here.
            """
        )

# Footer
st.markdown("---")
st.caption("Built with Streamlit and OpenAI API")
