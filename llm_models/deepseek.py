import os
from openai import OpenAI

def deepseek_chat(prompt):
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"      
    )
    response = client.chat.completions.create(
        model="deepseek-chat",  # **Model Name**
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content
