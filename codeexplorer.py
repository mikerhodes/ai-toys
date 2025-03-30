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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Code explorer tool")
parser.add_argument(
    "-n",
    "--num-turns",
    type=int,
    default=10,
    help="Number of turns (default: 10)",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="claude-3-7-sonnet-latest",
    help="Anthropic model (default: claude-3-7-sonnet-latest)",
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
pwd = Path(args.path).absolute()
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


def list_directory_tree(path: str) -> str:
    root = Path(path)
    if not root.exists():
        return f"ERROR: Path {root} does not exist"
    if not root.is_dir():
        return f"ERROR: Path {root} is not a directory"
    if root != jail and jail not in root.parents:
        return f"ERROR: Path {root} must have {jail} as an ancestor"

    result = [str(root.absolute().resolve()) + "/"]

    def _tree(dir_path: Path, prefix=""):
        paths = sorted(
            list(dir_path.iterdir()), key=lambda p: (not p.is_dir(), p.name)
        )

        count = len(paths)
        for i, path in enumerate(paths):
            if path.name.startswith("."):
                continue
            is_last = i == count - 1
            connector = "└── " if is_last else "├── "

            result.append(
                f"{prefix}{connector}{path.name}"
                + ("/" if path.is_dir() else "")
            )

            if path.is_dir():
                ext_prefix = prefix + ("    " if is_last else "│   ")
                _tree(path, ext_prefix)

    _tree(root)
    return "\n".join(result)


# memories = []


# def add_memory(memory: str) -> str:
#     memories.append(memory)
#     return "Added memory"


tools = [
    {
        "name": "print_working_directory",
        "description": """
            Return the absolute path of the working directory.
        """,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_directory_tree",
        "description": """
            List the complete file tree for path, including subdirectories.

            The passed path MUST be a directory, not a file.

            Use this tool to discover the files in the codebase. Once you know the files in the codebase, use the tree to construct the absolute paths to files.

            Here is an example tree, with a python project in a src directory:

            /full/path/to/folder/
            ├── src/
            │   └── mypackage/
            │       ├── entrypoint.py
            │       └── main.py
            ├── LICENSE
            ├── README.md
            ├── pyproject.toml
            └── uv.lock

            Real trees are likely to be much larger.
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
        case "print_working_directory":
            return print_working_directory()
        case "list_directory_tree":
            return list_directory_tree(tool_input["path"])
        case "read_file":
            return read_file(tool_input["path"])
        # case "add_memory":
        #     return add_memory(tool_input["memory"])
    return f"Error: no tool with name {tool_name}"


prompt = """
You are a programmer exploring a codebase. The aim is to find out what the program does and the key functions in the code base.

You have access to tools that tell you your working directory, list files, read files. Use these tools to explore the code in the directory.

You will initially be in the root directory of the project. Use the print_working_directory tool to find out the name of the directory. You must stay within this directory and not try to change to a directory outside this directory.

The best way to explore the code is to list the files in the directory using the list_directory_tree tool. Choose an important looking code file, like main.py, and read it. See if it tells you the purpose of the program. Read other files if needed to understand the purpose of the program. You might also look at README.md to understand the purpose of the program.

Once you've found the purpose of the program, read any additional files to find the important functions. Stop reading files once the purpose of the program is clear.

Once you have found the important functions, print out an explanation of the codebase:

- The purpose of the entire program.
- The key functions in the program.
"""

chat_history = []
num_turns = 0

console = Console()

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
        max_tokens=4096,
        messages=cast(Iterable[MessageParam], messages),
        tools=cast(Iterable[ToolUnionParam], tools),
    )

    # TODO need to strip out files read after they are memorised
    # if len(chat_history) > 8:
    #     chat_history = chat_history[-8:]

    print(f"Length of chat_history: {len(chat_history)}")

    print("\nResponse:")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content: {message.content}")

    if message.stop_reason == "tool_use":
        text_block = next(
            block for block in message.content if block.type == "text"
        )
        tool_use = next(
            block for block in message.content if block.type == "tool_use"
        )
        tool_name = tool_use.name
        tool_input = cast(Dict[str, str], tool_use.input)

        md = Markdown(
            tool_use_markdown.format(
                text=text_block.text,
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
        break

logger.info("Config: max turns: %d, path: %s", max_turns, jail)
logger.info("Took %d turns", num_turns)
logger.info("Used %s model", chat_model)
