import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, cast

import anthropic

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
    "path",
    nargs="?",
    default=".",
    help="Path to explore (default: current directory)",
)
args = parser.parse_args()

max_turns = args.num_turns


api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    logger.error("Set ANTHROPIC_API_KEY")
    exit(1)
client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-3-7-sonnet-latest"
MODEL = "claude-3-5-haiku-latest"

# Let Claude expore this folder for now
pwd = Path(args.path).absolute()
jail = pwd

logger.info("Config: turns: %d, path: %s", max_turns, jail)


#
# Tools
#


def print_working_directory() -> str:
    return str(pwd.absolute())


def list_working_directory() -> str:
    result = []
    for p in pwd.iterdir():
        if p.name.startswith("."):
            continue
        result.append(
            {"name": p.name, "type": "directory" if p.is_dir() else "file"}
        )
    return json.dumps(result)


def change_working_directory(cd: str) -> str:
    global pwd
    d = Path(cd).absolute()
    if not d.exists():
        return f"ERROR: Path {cd} does not exist; working directory is still {pwd}"
    if jail not in d.parents:
        return f"ERROR: Path {cd} must have {jail} as an ancestor"
    pwd = d
    return f"Working directory is now {pwd}"


def read_file(fpath: str) -> str:
    p = Path(fpath).absolute()
    if not p.exists():
        return f"ERROR: Path {fpath} does not exist"
    if jail not in p.parents:
        return f"ERROR: Path {p} must have {jail} as an ancestor"
    with open(fpath, "r") as f:
        return f.read()


memories = []


def add_memory(memory: str) -> str:
    memories.append(memory)
    return "Added memory"


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
        "name": "list_working_directory",
        "description": """
            List the entries in the working directory.

            Entries will be returned as JSON:

            [{"type": "file", "name":"Makefile"}, {"type": "directory", "name":"src"}, ]
        """,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "change_working_directory",
        "description": """
            Change the working directory.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path for the new working directory",
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
    {
        "name": "add_memory",
        "description": """
            Add a memory about the code base. The data in memory will be sent back to you in your prompt. To avoid calling tools again and again, use the memory to note down information about the results of each step.

            Examples:

            <memory>
            main.py: this contains the functions main, read_foo, write_bar. main contains the main program code. read_foo reads the foo file from disk. write_bar writes to the cloud service bar.
            </memory>

            <memory>
            the files in /the/root/path are main.py, helper.py, readme.md
            </memory>
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "Memory to add",
                }
            },
            "required": ["memory"],
        },
    },
]


def process_tool_call(tool_name: str, tool_input: Dict[str, str]) -> str:
    match tool_name:
        case "print_working_directory":
            return print_working_directory()
        case "change_working_directory":
            return change_working_directory(tool_input["path"])
        case "list_working_directory":
            return list_working_directory()
        case "read_file":
            return read_file(tool_input["path"])
        case "add_memory":
            return add_memory(tool_input["memory"])
    return f"Error: no tool with name {tool_name}"


prompt = """
You are a programmer exploring a codebase. The aim is to find out what the program does and the key functions in the code base.

You have access to tools that tell you your working directory, list files, change working directory and read files. Use these tools to explore the code in the directory.

You will initially be in the root directory of the project. Use the print_working_directory tool to find out the name of the directory. You must stay within this directory and not try to change to a directory outside this directory.

The best way to explore the code is to list the files in the directory using the list_working_directory tool. Choose an important looking code file, like main.py, and read it. See if it tells you the purpose of the program. Read other files if needed to understand the purpose of the program. You might also look at README.md to understand the purpose of the program.

Once you've found the purpose of the program, read any additional files to find the important functions.

Every time you call a tool, use the add_memory tool to add a memory if the information will be needed again. This is very important because the tool response will not be included in future turns! Use the add_memory tool right after getting the response to another tool.

Examples of things you will need to remember:

- Remember the root directory.
- Remember the content of directories after the list_directory_content tool. Remember to include the directory path in the memory.
- Summarise files, including information about the functions in the files that seem important.

For example:

<memory>
main.py: this contains the functions main, read_foo, write_bar. main contains the main program code. read_foo reads the foo file from disk. write_bar writes to the cloud service bar.
</memory>

<memory>
/home/mike/code/app contents are main.py, README.md, foo.txt, processor.py
</memory>
 
Once you have found the important functions, print out an explanation of the codebase:

- The purpose of the entire program.
- The key functions in the program.
"""
prompt_message = {"role": "user", "content": prompt}
print(f"\n{'=' * 50}\nInitial prompt: {prompt}\n{'=' * 50}")

chat_history = []

for i in range(0, max_turns):
    print(f"\n{'=' * 50}")

    memories_message = "\n".join([f"<memory>{x}</memory>" for x in memories])

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            prompt_message,
            {"role": "user", "content": memories_message},
        ]
        + chat_history,
        tools=tools,
    )

    # TODO need to strip out files read after they are memorised
    if len(chat_history) > 8:
        chat_history = chat_history[-8:]

    print(f"Length of chat_history: {len(chat_history)}")

    print(f"\nResponse:")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content: {message.content}")

    if message.stop_reason == "tool_use":
        tool_use = next(
            block for block in message.content if block.type == "tool_use"
        )
        tool_name = tool_use.name
        tool_input = cast(Dict[str, str], tool_use.input)

        print(f"\nTool Used: {tool_name}")
        print(f"Tool Input: {tool_input}")

        tool_result = process_tool_call(tool_name, tool_input)

        print(f"Tool Result: {tool_result[:200]}")

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
        print(message.content)
        break
