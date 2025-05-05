"""
codeexplorer.py

Implements an agent-model to allow LLM to explore a codebase
using tools, rather than trying to pre-create a large context
from the codebase ourselves.
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

logging.basicConfig(level=logging.WARNING)
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
    default="qwen3:8b",
    help="Model (default: qwen3:8b)",
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

ollama_client = ollama.Client()

# Let model expore this folder for now
pwd = Path(args.path).resolve()
jail = pwd

logger.info(
    "Config: turns: %d, path: %s, model: %s", max_turns, jail, chat_model
)


#
# Tools
#


def read_file_path(path: str) -> str:
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
        "parameters": {
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
        "name": "read_file_path",
        "description": """
            Read a file.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file",
                }
            },
            "required": ["path"],
        },
    },
]

wxtools = [{"type": "function", "function": x} for x in tools]


def process_tool_call(tool_name: str, tool_input: Dict[str, str]) -> str:
    match tool_name:
        case "list_directory_simple":
            return list_directory_simple(tool_input["path"])
        case "read_file_path":
            return read_file_path(tool_input["path"])
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

Tool Result:
```
{tool_result}
```
"""


for i in range(0, max_turns):
    num_turns += 1
    logger.debug(f"\n{'=' * 50}")

    cache_prompt = None
    if chat_history:
        cache_prompt = copy.deepcopy(chat_history[-1])

    messages = (
        [
            {"role": "user", "content": prompt},
        ]
        + chat_history[:-1]
        + ([cache_prompt] if cache_prompt else [])
    )

    logger.debug("Messaging model")
    message = ollama_client.chat(
        model=chat_model,
        messages=messages,
        tools=wxtools,
        options=ollama.Options(
            num_ctx=16384,
        ),
    )

    logger.debug(f"Length of chat_history: {len(chat_history)}")

    choice = message

    logger.debug("\nResponse:")
    logger.debug(f"Stop Reason: {choice['done_reason']}")
    logger.debug(f"Content: {choice['message']}")

    if choice.message.tool_calls:
        tool_call = choice["message"]["tool_calls"][0]
        f = tool_call["function"]
        tool_name = f["name"]
        tool_input = f["arguments"]

        tool_result = process_tool_call(tool_name, tool_input)

        md = Markdown(
            tool_use_markdown.format(
                text="",  # text_block.text if text_block else "",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result="\n".join(
                    tool_result.split("\n")[:10] + ["..."]
                ),
            )
        )
        console.print(Panel(md, title="Turn"))

        chat_history.append(
            {
                "role": "tool",
                "content": tool_result,
                "name": tool_name,
            }
        )
    else:
        chat_history.append(
            {
                "role": "assistant",
                "content": choice.message.content,
            }
        )

        message = (
            choice["message"]["content"]
            .replace("<think>", "`<think>`")
            .replace("</think>", "`</think>`")
        )
        md = Markdown(message)
        console.print(Panel(md, title="Code exploration result"))
        if args.output:
            with open(
                Path(args.output).absolute(), "w", encoding="utf-8"
            ) as f:
                f.write(message)
        break

logger.info("Config: max turns: %d, path: %s", max_turns, jail)
logger.info("Took %d turns", num_turns)
logger.info("Used %s model", chat_model)
