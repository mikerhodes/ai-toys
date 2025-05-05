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
from typing import Dict, Any, Tuple

import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt


class OllamaAdapter:
    def __init__(self):
        self.ollama_client = ollama.Client()

    def prepare_messages(self, chat_history):
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
        return messages

    def chat(self, **kwargs):
        # message = ollama_client.chat(
        #     model=chat_model,
        #     messages=messages,
        #     tools=openai_tools,
        #     options=ollama.Options(
        #         num_ctx=16384,
        #     ),
        # )
        chat_response = self.ollama_client.chat(
            options=ollama.Options(
                num_ctx=16384,
            ),
            **kwargs,
        )
        logger.debug("\nResponse:")
        logger.debug(f"Stop Reason: {chat_response['done_reason']}")
        logger.debug(f"Content: {chat_response['message']}")
        return chat_response

    def get_response_text(self, chat_response) -> str:
        return chat_response.message.content

    def tools_for_model(self, openai_tools):
        return openai_tools

    def has_tool_use(self, chat_response):
        return bool(chat_response.message.tool_calls)

    def get_tool_use(self, chat_response) -> Tuple[str, Dict[str, str], str]:
        tool_call = chat_response["message"]["tool_calls"][0]
        f = tool_call["function"]
        tool_name = f["name"]
        tool_input = f["arguments"]
        return tool_name, tool_input, ""

    def format_assistant_history_message(self, chat_response):
        return {
            "role": "assistant",
            "content": chat_response.message.content,
        }

    def format_tool_result_message(
        self, tool_name: str, tool_use_id: str, tool_result: str
    ) -> Dict:
        return {
            "role": "tool",
            "content": tool_result,
            "name": tool_name,
        }


client = OllamaAdapter()

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
openai_tools = [{"type": "function", "function": x} for x in tools]


def process_tool_call(tool_name: str, tool_input: Dict[str, str]) -> str:
    match tool_name:
        case "list_directory_simple":
            return list_directory_simple(tool_input["path"])
        case "read_file_path":
            return read_file_path(tool_input["path"])
    return f"Error: no tool with name {tool_name}"


console = Console()

if not args.prompt:
    extras = Prompt.ask(
        "Anything you'd like AI to think about (eg, plan how to do X)",
        console=console,
    )
    if extras:
        user_prompt = extras
    else:
        user_prompt = "Please explain this codebase"
else:
    user_prompt = args.prompt

prompt = f"""
You are a programmer exploring a codebase.

You have access to tools that tell you your working directory, list files, read files. Use these tools to explore the code in the directory.

The best way to explore the code is to list the files in the directory using the list_directory_simple tool. It will return a list of all files in the directory tree, including subdirectories.

Choose some important looking code files in the source tree and use the the read_file_path tool to read the files. Read more files if needed to understand the purpose of the program.

If the read_file_path tool fails, double check the path you passed in!

Take your time and be sure you've looked at everything you need to understand the program and answer the user's question below.

The project root directory is: {Path(args.path).resolve()}

Here's the user's question:

{user_prompt}
"""

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

chat_history = []
num_turns = 0

for i in range(0, max_turns):
    num_turns += 1
    logger.debug(f"\n{'=' * 50}")

    messages = client.prepare_messages(chat_history)

    logger.debug("Messaging model")
    chat_response = client.chat(
        model=chat_model,
        messages=messages,
        tools=client.tools_for_model(openai_tools),
    )

    logger.debug(f"Length of chat_history: {len(chat_history)}")

    if client.has_tool_use(chat_response):
        response_text = client.get_response_text(chat_response)
        tool_name, tool_input, tool_use_id = client.get_tool_use(
            chat_response
        )
        tool_result = process_tool_call(tool_name, tool_input)

        md = Markdown(
            tool_use_markdown.format(
                text=response_text,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result="\n".join(
                    tool_result.split("\n")[:10] + ["..."]
                ),
            )
        )
        console.print(Panel(md, title="Turn"))

        chat_history.append(
            client.format_assistant_history_message(chat_response)
        )
        chat_history.append(
            client.format_tool_result_message(
                tool_name, tool_use_id, tool_result
            )
        )
    else:
        chat_history.append(
            client.format_assistant_history_message(chat_response)
        )
        message_text = (
            client.get_response_text(chat_response)
            .replace("<think>", "`<think>`")
            .replace("</think>", "`</think>`")
        )
        md = Markdown(message_text)
        console.print(Panel(md, title="Code exploration result"))
        if args.output:
            with open(
                Path(args.output).absolute(), "w", encoding="utf-8"
            ) as f:
                f.write(message_text)
        break

logger.info("Config: max turns: %d, path: %s", max_turns, jail)
logger.info("Took %d turns", num_turns)
logger.info("Used %s model", chat_model)
