import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

load_dotenv()


PROMPT = "Reply with exactly one short sentence confirming you are working."

def test_gemini():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
    response = model.generate_content(PROMPT)
    return response.text.strip()

def test_groq():
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": PROMPT}],
    )
    return response.choices[0].message.content.strip()


def test_openrouter():
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": PROMPT}],
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    tests = {
        "Gemini": test_gemini,
        "Groq": test_groq,
        "OpenRouter": test_openrouter,
    }
    for name, fn in tests.items():
        print(f"\n--- Testing {name} ---")
        try:
            result = fn()
            print(f"✅ {name} responded: {result}")
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")