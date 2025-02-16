from google import genai
import os 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") )

def gemini_api(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    print(gemini_api("What date is it today?"))
