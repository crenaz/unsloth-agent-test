
import os
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["UNSLOTH_BASE_URL"],
    api_key=os.environ["UNSLOTH_API_KEY"],
)

# Restrict this test tool to /tmp.
ALLOWED_DIRECTORY = Path("/tmp").resolve()
MAX_ENTRIES = 100


def list_files(directory: str) -> dict:
"""     requested = Path(directory).resolve()

    if requested != ALLOWED_DIRECTORY:
        return {
            "error": "This test only permits listing /tmp."
        } """
# Accept "tmp" or "/tmp", but nothing else.
if directory not in ("tmp", "/tmp"):
    return {
        "error": "This test only permits listing /tmp."
    }

requested = ALLOWED_DIRECTORY

    entries = []

    try:
        with os.scandir(ALLOWED_DIRECTORY) as iterator:
            for entry in iterator:
                if len(entries) >= MAX_ENTRIES:
                    return {
                        "directory": str(ALLOWED_DIRECTORY),
                        "entries": entries,
                        "truncated": True,
                    }

                try:
                    if entry.is_dir(follow_symlinks=False):
                        entry_type = "directory"
                    elif entry.is_file(follow_symlinks=False):
                        entry_type = "file"
                    else:
                        entry_type = "other"

                    entries.append({
                        "name": entry.name,
                        "type": entry_type,
                    })

                except OSError as exc:
                    entries.append({
                        "name": entry.name,
                        "error": str(exc),
                    })

    except OSError as exc:
        return {"error": str(exc)}

    return {
        "directory": str(ALLOWED_DIRECTORY),
        "entries": entries,
        "truncated": False,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in /tmp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The directory to inspect. Must be /tmp.",
                    }
                },
                "required": ["directory"],
                "additionalProperties": False,
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a software engineering assistant. "
            "When asked to list files, you must use the list_files tool. "
            "Pass the directory argument exactly as '/tmp'. "
            "After receiving the tool result, summarize the actual result. "
            "Never invent file names."
        ),
    },
    {
        "role": "user",
        "content": "Use the list_files tool to list the contents of /tmp.",
    },
]

# 1. Ask GPT-OSS to request the tool.
print("\n=== STEP 1: REQUESTING TOOL CALL ===")

response = client.chat.completions.create(
    model="unsloth/gpt-oss-20b-GGUF",
    messages=messages,
    tools=tools,
    tool_choice="required",
    parallel_tool_calls=False,
    temperature=0,
    max_tokens=512,
)

message = response.choices[0].message

if not message.tool_calls:
    raise RuntimeError(
        "GPT-OSS did not return a structured tool call."
    )

# Preserve the assistant's structured tool-call message.
messages.append({
    "role": "assistant",
    "content": message.content,
    "tool_calls": [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in message.tool_calls
    ],
})

# 2. Execute the requested function locally.
print("\n=== STEP 2: EXECUTING LOCAL TOOL ===")

for call in message.tool_calls:
    if call.function.name != "list_files":
        raise RuntimeError(
            f"Unexpected tool requested: {call.function.name}"
        )

    arguments = json.loads(call.function.arguments)
    print("Tool:", call.function.name)
    print("Arguments:", json.dumps(arguments, indent=2))

    result = list_files(**arguments)

    print("\nActual local result:")
    print(json.dumps(result, indent=2))

    # 3. Send the real execution result back to the model.
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    })

# 4. Ask GPT-OSS to interpret the returned tool result.
print("\n=== STEP 3: SENDING TOOL RESULT BACK TO GPT-OSS ===")

final_response = client.chat.completions.create(
    model="unsloth/gpt-oss-20b-GGUF",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=False,
    temperature=0,
    max_tokens=512,
)

final_message = final_response.choices[0].message

print("\n=== STEP 4: FINAL MODEL RESPONSE ===")
print(final_message.content or "(No text response returned)")

if final_message.tool_calls:
    print("\nNote: The model requested another tool call.")
    for call in final_message.tool_calls:
        print(call.function.name, call.function.arguments)

print("\n=== ROUND-TRIP TEST COMPLETED ===")
