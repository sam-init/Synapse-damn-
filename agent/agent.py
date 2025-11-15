import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def run_agent(form_data, image_data, question=None):
    """
    form_data: dict (OCR fields)
    image_data: dict (CNN output)
    question: optional string from user
    """

    url = "https://api.groq.com/openai/v1/chat/completions"

    system_prompt = """
You are the FinSecure Claim Triage Reasoning Agent.

You always respond in normal, clear English sentences.
Never output JSON unless explicitly asked.

You are given:
1. OCR extracted claim form fields:
   - Policy number
   - Incident date
   - Incident description
2. Computer vision damage analysis:
   - Detected damage severity
   - Confidence level
3. The user may ask follow-up questions.

Your tasks:
- Summarize all provided data.
- Identify consistency or mismatch between the claim description and the detected damage.
- Detect fraud indicators if present.
- Provide a triage decision using these categories:
    • Auto-Approve
    • Flag for Review
    • High Priority
    • Fraud Risk
- Explain the reasoning step-by-step in clean, natural language.
- If the user asks a question, answer conversationally using the claim details.
"""

    # Combine all data into a natural-readable user message
    user_message = f"""
The OCR system extracted the following fields:
{json.dumps(form_data, indent=2)}

The CNN damage analysis produced:
{json.dumps(image_data, indent=2)}

User question:
{question if question else "No follow-up question provided."}
"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
