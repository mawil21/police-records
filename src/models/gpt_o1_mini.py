from openai import OpenAI

client = OpenAI()


def chatGPT_api(message_content):
    response = client.chat.completions.create(
        model = "o1-mini",
        messages = [
            {"role": "user", "content": message_content}],
  
    )

    return response.choices[0].message.content


def gpt_o1_preview(prompt):
    message_content = prompt[0] + prompt[1]
    return chatGPT_api(message_content)