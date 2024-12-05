from openai import OpenAI

client = OpenAI()

def chatGPT_api(message_content,temperature=0):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {"role": "user", "content": message_content}],
        temperature = temperature
    )

    return response.choices[0].message.content


def gpt_4o(prompt):
    message_content = prompt[0] + prompt[1]
    return chatGPT_api(message_content)