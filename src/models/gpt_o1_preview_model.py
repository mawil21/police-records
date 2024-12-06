from openai import OpenAI

client = OpenAI()

def chatGPT_api(message_content):
    response = client.chat.completions.create(
        model = "o1-preview",
        response_format={'type':'json_object'},
        messages = [
            {"role" : "user", "content" : "Provide output in valid JSON."},
            {"role": "user", "content": message_content}
        ]
    )
    return response.choices[0].message.content

def gpt_o1_preview(prompt):
    return chatGPT_api(prompt)