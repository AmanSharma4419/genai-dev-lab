import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ----------------------------
# TOOLS
# ----------------------------

def get_current_location():
    # Dummy example
    return {
        "city": "Dharamshala"
    }


def get_weather(city):
    url = f"https://wttr.in/{city}?format=j1"
    res = requests.get(url)
    data = res.json()

    current = data["current_condition"][0]

    return {
        "city": city,
        "temp": current["temp_C"],
        "humidity": current["humidity"],
        "condition": current["weatherDesc"][0]["value"]
    }


# ----------------------------
# SYSTEM PROMPT
# ----------------------------

system_prompt = """
You are an AI assistant.

You have 2 tools:

1. get_weather(city)
Use when user asks weather of a city.

2. get_current_location()
Use when user asks weather of current location like:
- weather here
- weather near me
- my location weather

Return ONLY JSON:

For weather:
{
 "tool":"weather",
 "city":"city_name"
}

For location:
{
 "tool":"location"
}
"""

# ----------------------------
# LOOP
# ----------------------------

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

    tool = parsed["tool"]

    # TOOL 1: LOCATION
    if tool == "location":
        loc = get_current_location()
        city = loc["city"]

        weather = get_weather(city)

        print(f"""
Weather in {weather['city']}
Temperature: {weather['temp']}°C
Humidity: {weather['humidity']}%
Condition: {weather['condition']}
""")

    # TOOL 2: WEATHER DIRECT
    elif tool == "weather":
        city = parsed["city"]
        weather = get_weather(city)

        print(f"""
Weather in {weather['city']}
Temperature: {weather['temp']}°C
Humidity: {weather['humidity']}%
Condition: {weather['condition']}
""")