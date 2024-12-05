from openai import OpenAI

client = OpenAI()

def chatGPT_api(message_content,temperature=0):
    response = client.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "user", "content": message_content}],
        temperature = temperature
    )

    return response.choices[0].message.content

def gpt_4(prompt):
    return chatGPT_api(prompt)