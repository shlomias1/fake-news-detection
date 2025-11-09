from openai import OpenAI
client = OpenAI()  # ישלוף את OPENAI_API_KEY מהסביבה

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers concisely."},
        {"role": "user", "content": "תן 4 עובדות קצרות על מערכת השמש"}
    ]
)
print(resp.choices[0].message.content)