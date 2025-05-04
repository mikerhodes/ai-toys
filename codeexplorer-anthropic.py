"""
codeexplorer.py

Implements an agent-model to allow LLM to explore a codebase
using tools, rather than trying to pre-create a large context
from the codebase ourselves.
"""

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, cast

import anthropic
from anthropic.types import MessageParam, ToolUnionParam
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Code explorer tool")
parser.add_argument(
    "-n",
    "--num-turns",
    type=int,
    default=20,
    help="Number of turns (default: 20)",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="claude-3-7-sonnet-latest",
    help="Anthropic model (default: claude-3-7-sonnet-latest)",
)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default=None,
    help="Question to answer using codebase (default: prompt user for question)",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Write final output to file (default: write only to terminal)",
)
parser.add_argument(
    "path",
    nargs="?",
    default=".",
    help="Path to explore (default: current directory)",
)
args = parser.parse_args()

max_turns = args.num_turns
chat_model = args.model

# MODEL = "claude-3-7-sonnet-latest"
# MODEL = "claude-3-5-haiku-latest"

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    logger.error("Set ANTHROPIC_API_KEY")
    exit(1)
client = anthropic.Anthropic(api_key=api_key)

# Let Claude expore this folder for now
pwd = Path(args.path).resolve()
jail = pwd

logger.info(
    "Config: turns: %d, path: %s, model: %s", max_turns, jail, chat_model
)


#
# Tools
#


def print_working_directory() -> str:
    return str(pwd.absolute())


def read_file(path: str) -> str:
    p = Path(path).absolute()
    if not p.exists():
        return f"ERROR: Path {path} does not exist"
    if jail not in p.parents:
        return f"ERROR: Path {p} must have {jail} as an ancestor"
    with open(path, "r") as f:
        return f.read()


def list_directory_simple(path: str) -> str:
    """
    Return a list of files in the directory and subdirectories as a
    list of absolute file paths, one path per line.
    """
    root = Path(path)
    if not root.exists():
        return f"ERROR: Path {root} does not exist"
    if not root.is_dir():
        return f"ERROR: Path {root} is not a directory"
    if root != jail and jail not in root.parents:
        return f"ERROR: Path {root} must have {jail} as an ancestor"

    result = []

    def _tree(dir_path: Path):
        for path in [x for x in dir_path.iterdir() if x.is_file()]:
            if path.name.startswith("."):
                continue
            result.append(str(path.absolute()))

        for path in [x for x in dir_path.iterdir() if x.is_dir()]:
            if path.name.startswith("."):
                continue
            if path.is_dir():
                _tree(path)

    _tree(root)
    result = sorted(result)
    return "\n".join(result)


# memories = []


# def add_memory(memory: str) -> str:
#     memories.append(memory)
#     return "Added memory"


tools = [
    {
        "name": "list_directory_simple",
        "description": """
            List the complete file tree for path, including subdirectories.

            The passed path MUST be a directory, not a file.

            Use this tool to discover the files in the codebase. The returned value is a list of absolute paths. Use these paths to explore the codebase.

            Here is an example tree, with a python project in a src directory:

            /Users/mike/code/pythonapp/src/mypackage/entrypoint.py
            /Users/mike/code/pythonapp/src/mypackage/main.py
            /Users/mike/code/pythonapp/LICENSE
            /Users/mike/code/pythonapp/README.md
            /Users/mike/code/pythonapp/pyproject.toml
            /Users/mike/code/pythonapp/uv.lock

            Real lists are likely to be much larger!
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to list files",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": """
            Read a file.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file to read",
                }
            },
            "required": ["path"],
        },
    },
    # {
    #     "name": "add_memory",
    #     "description": """
    #         Add a memory about the code base. The data in memory will be sent back to you in your prompt. To avoid calling tools again and again, use the memory to note down information about the results of each step.
    #         Examples:
    #         <memory>
    #         main.py: this contains the functions main, read_foo, write_bar. main contains the main program code. read_foo reads the foo file from disk. write_bar writes to the cloud service bar.
    #         </memory>
    #         <memory>
    #         the files in /the/root/path are main.py, helper.py, readme.md
    #         </memory>
    #     """,
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "memory": {
    #                 "type": "string",
    #                 "description": "Memory to add",
    #             }
    #         },
    #         "required": ["memory"],
    #     },
    # },
]


def process_tool_call(tool_name: str, tool_input: Dict[str, str]) -> str:
    match tool_name:
        case "list_directory_simple":
            return list_directory_simple(tool_input["path"])
        case "read_file":
            return read_file(tool_input["path"])
        # case "add_memory":
        #     return add_memory(tool_input["memory"])
    return f"Error: no tool with name {tool_name}"


prompt = f"""
You are a programmer exploring a codebase.

You have access to tools that tell you your working directory, list files, read files. Use these tools to explore the code in the directory.

The best way to explore the code is to list the files in the directory using the list_directory_simple tool. It will return a list of all files in the directory tree, including subdirectories.

Choose some important looking code files in the source tree and use the the read_file_path tool to read the files. Read more files if needed to understand the purpose of the program.

If the read_file_path tool fails, double check the path you passed in!

Take your time and be sure you've looked at everything you need to understand the program and answer the user's question below.

The project root directory is: {Path(args.path).resolve()}

Here's the user's question:
"""

# TODO Could we just put the root folder into the prompt rather than
# making the model ask for it?

chat_history = []
num_turns = 0


console = Console()


if not args.prompt:
    extras = Prompt.ask(
        "Anything you'd like AI to think about (eg, plan how to do X)",
        console=console,
    )

    if extras:
        prompt = "\n\n".join([prompt, extras])
    else:
        # create a default question
        prompt = "\n\n".join([prompt, "Please explain this codebase"])
else:
    prompt = "\n\n".join([prompt, args.prompt])

console.print(Panel(Markdown(prompt), title="Prompt"))

# dedented markdown to use when formatting each tool use message
tool_use_markdown = """
{text}

Tool Used: `{tool_name}`

Tool Input: `{tool_input}`
"""

for i in range(0, max_turns):
    num_turns += 1
    print(f"\n{'=' * 50}")

    # memories_message = "\n".join([f"<memory>{x}</memory>" for x in memories])

    # For each turn, we want to tell the Anthropic API that we
    # want to cache up to this point, so we can reuse it on the
    # next turn. We don't want to end up with the cache directive
    # in every turn, however, so create a copy of the last message
    # and add the cache control to it --- this ensures that we only
    # ever have the cache control block in the latest message.
    cache_prompt = None
    if chat_history:
        cache_prompt = copy.deepcopy(chat_history[-1])
        cache_prompt["content"][0]["cache_control"] = {"type": "ephemeral"}

    messages = (
        [
            {"role": "user", "content": prompt},
            # {"role": "user", "content": memories_message},
        ]
        + chat_history[:-1]
        + ([cache_prompt] if cache_prompt else [])
    )

    message = client.messages.create(
        model=chat_model,
        max_tokens=8192,
        messages=cast(Iterable[MessageParam], messages),
        tools=cast(Iterable[ToolUnionParam], tools),
    )

    logger.debug(
        "cache_creation_input_tokens: %d, cache_read_input_tokens: %d, input_tokens: %d",
        message.usage.cache_creation_input_tokens,
        message.usage.cache_read_input_tokens,
        message.usage.input_tokens,
    )

    # TODO need to strip out files read after they are memorised
    # if len(chat_history) > 8:
    #     chat_history = chat_history[-8:]

    logger.debug(f"Length of chat_history: {len(chat_history)}")

    logger.debug("\nResponse:")
    logger.debug(f"Stop Reason: {message.stop_reason}")
    logger.debug(f"Content: {message.content}")

    if message.stop_reason == "tool_use":
        try:
            text_block = next(
                block for block in message.content if block.type == "text"
            )
        except StopIteration:
            text_block = None  # sometimes the model has nothing to say
        tool_use = next(
            block for block in message.content if block.type == "tool_use"
        )
        tool_name = tool_use.name
        tool_input = cast(Dict[str, str], tool_use.input)

        md = Markdown(
            tool_use_markdown.format(
                text=text_block.text if text_block else "",
                tool_name=tool_name,
                tool_input=tool_input,
            )
        )
        console.print(Panel(md, title="Turn"))

        # print(f"\nTool Used: {tool_name}")
        # print(f"Tool Input: {tool_input}")

        tool_result = process_tool_call(tool_name, tool_input)

        # print(f"Tool Result: {tool_result[:200]}")

        chat_history.append(
            {"role": "assistant", "content": message.content}
        )
        chat_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result,
                    }
                ],
            }
        )
    else:
        chat_history.append(
            {"role": "assistant", "content": message.content}
        )
        text_block = next(
            block for block in message.content if block.type == "text"
        )

        md = Markdown(text_block.text)
        console.print(Panel(md, title="Code exploration result"))
        if args.output:
            with open(
                Path(args.output).absolute(), "w", encoding="utf-8"
            ) as f:
                f.write(text_block.text)
        break

logger.info("Config: max turns: %d, path: %s", max_turns, jail)
logger.info("Took %d turns", num_turns)
logger.info("Used %s model", chat_model)
