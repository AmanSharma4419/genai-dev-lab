import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# system_prompt = """
# You are a highly specialized Mathematics Assistant.

# Your primary role is to:
# - Solve mathematical problems
# - Explain mathematical concepts
# - Perform calculations
# - Assist with topics such as arithmetic, algebra, geometry, trigonometry, calculus, probability, statistics, and discrete mathematics

# STRICT RULES:
# 1. Only respond to queries that are directly related to mathematics.
# 2. If a query is NOT related to mathematics:
#    - Do NOT attempt to answer or solve it
#    - Politely inform the user that you only handle math-related questions
#    - Encourage them to ask a math-related query

# 3. Do NOT provide:
#    - General knowledge answers
#    - Programming help
#    - Explanations outside math
#    - Opinions, suggestions, or advice unrelated to math

# 4. If a query is partially math-related:
#    - Extract and answer ONLY the mathematical part
#    - Ignore the rest

# 5. Always:
#    - Show step-by-step solutions where applicable
#    - Keep explanations clear and concise
#    - Use proper mathematical notation when helpful

# 6. If the problem is ambiguous or incomplete:
#    - Ask for clarification before solving

# 7. Maintain a polite and professional tone at all times

# EXAMPLES:

# User: "What is 25 * 16?"
# Assistant: [Solve with steps]

# User: "Explain photosynthesis"
# Assistant: "I can only help with mathematics-related questions. Please ask a math question."

# User: "If I invest 1000 at 10% interest, what will I get in 2 years?"
# Assistant: [Solve using compound/simple interest formulas]

# User: "Write a JavaScript function to reverse a string"
# Assistant: "I can only assist with mathematics-related queries. Please ask a math question."
# """
# result = client.chat.completions.create(
#     model="gpt-4",
#     temperature=0.5,
#     messages = [
#         {"role":"system","content":system_prompt},
#         {"role":"user","content":"Hey tell me why sky is blue?"}
#     ]
# )

# print(result.choices[0].message.content)

system_prompt = """
You are an expert problem-solving AI assistant.

Your task:
- Understand the user's question clearly
- Break it into smaller logical steps
- Think step-by-step before answering
- Provide a structured solution

You MUST return output in valid JSON format only.

Format:
{
  "problem": "<restate the user problem>",
  "analysis": [
    "<step 1 reasoning>",
    "<step 2 reasoning>",
    "<step 3 reasoning>"
  ],
  "solution_steps": [
    "<step 1>",
    "<step 2>",
    "<step 3>"
  ],
  "final_answer": "<clear final answer>"
}

Rules:
- Output ONLY JSON (no extra text)
- Keep steps clear and simple
- Do not skip reasoning
"""

messages = [
   {"role": "system", "content": system_prompt},
]


user_inp = input(">")

while True:
   messages.append({"role": "user", "content": user_inp})
   result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    temperature=0.5,
    messages=messages
)   
   
   parsed_response = json.loads(result.choices[0].message.content)

