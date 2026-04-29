import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def run_cmd(cmd):
    result = os.system(cmd)   # or subprocess for better output
    return result


system_prompt = """
You are a shell command generator.

Convert the user's request into the exact terminal command.

Return ONLY JSON:

{
  "cmd": "command_here"
}

Rules:
- Return only JSON
- Safe commands only
- Prefer Linux/macOS commands
- If unclear:
{
  "cmd":"clarify"
}
"""


while True:
    user_inp = input("> ")

    result = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_inp}
        ]
    )

    parsed = json.loads(result.choices[0].message.content)

    cmd = parsed["cmd"]

    if cmd == "clarify":
        print("Please clarify request")
        continue

    if cmd == "refused":
        print("Request refused")
        continue

    print("Running:", cmd)

    output = run_cmd(cmd)

    print("Done:", output)