import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()


system_prompt = """
You are a weather agent.

If user asks weather related query,
return JSON only:

{
 "tool":"weather",
 "city":"city_name"
}
"""

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

while True: 
    user_inp = input("> ")
    result = client.chat.completions.create(
             model="gpt-4o",
             response_format={"type":"json_object"},
             messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_inp}
        ]
    )   
    
    parsed = json.loads(result.choices[0].message.content)
    print(parsed,"parsed")
    city = parsed["city"]
    weather = get_weather(city)
    print(f"""
Weather in {weather['city']}
Temperature: {weather['temp']}°C
Humidity: {weather['humidity']}%
Condition: {weather['condition']}
""")