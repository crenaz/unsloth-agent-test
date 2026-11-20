
import os
import json
from openai import OpenAI

# Connect to Unsloth Studio's OpenAI-compatible API
client = OpenAI(
    base_url=os.environ["UNSLOTH_BASE_URL"],
    api_key=os.environ["UNSLOTH_API_KEY"],
)

# Define a function the model can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a given directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Absolute path of the directory to inspect."
                    }
                },
                "required": ["directory"],
                "additionalProperties": False
            }
        }
    }
]

# Send a request requiring a function call
response = client.chat.completions.create(
    model="unsloth/gpt-oss-20b-GGUF",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a software engineering assistant. "
                "When a user requests a directory listing, "
                "you must call the list_files function. "
                "Do not merely describe the function call."
            )
        },
        {
            "role": "user",
            "content": "Use the list_files tool to list all files in /tmp."
        }
    ],
    tools=tools,
    tool_choice="required",
    parallel_tool_calls=False,
    temperature=0,
    max_tokens=256,
)

message = response.choices[0].message

# Display the complete assistant response
print("\n=== RAW ASSISTANT MESSAGE ===")
print(message.model_dump_json(indent=2))

# Inspect the structured tool calls
print("\n=== STRUCTURED TOOL CALLS ===")

if message.tool_calls:
    for call in message.tool_calls:
        print("Function:", call.function.name)
        print(
            "Arguments:",
            json.dumps(
                json.loads(call.function.arguments),
                indent=2
            )
        )
else:
    print("NO STRUCTURED TOOL CALLS RETURNED")
    print("Assistant content:", message.content)
