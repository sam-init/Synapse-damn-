import base64
import mimetypes
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image
import os

# ---------------------------------------------------
# LOAD env
# ---------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


# ---------------------------------------------------
# BASE64 HELPERS
# ---------------------------------------------------
def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def pdf_to_base64(path):
    return file_to_base64(path)

def image_to_base64(path):
    return file_to_base64(path)


# ---------------------------------------------------
# WHAT TYPE IS THE FILE?
# ---------------------------------------------------
def detect_file_type(path):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        ext = path.lower().split(".")[-1]
        if ext in ("jpg", "jpeg", "png"):
            mime = f"image/{ext}"
        elif ext == "pdf":
            mime = "application/pdf"
    return mime


# ---------------------------------------------------
# BUILD PROMPT
# ---------------------------------------------------
EXTRACTION_PROMPT = """
You are an expert document-understanding system.

Extract ONLY the following fields from the uploaded document:

{
  "policy_number": "",
  "incident_date": "",
  "incident_description": ""
}

Rules:
- Decode handwriting fully.
- Extract policy number exactly as written.
- Extract date of loss in DD/MM/YYYY.
- Extract the handwritten cause-of-loss.
- If multiple pages exist, search all pages.
- Return only pure JSON. No commentary.
"""


# ---------------------------------------------------
# MASTER FUNCTION: PDF + IMAGE SUPPORT
# ---------------------------------------------------
def extract_claim_fields(file_path):
    mime = detect_file_type(file_path)

    if mime is None:
        raise ValueError("Unknown file type.")

    print(f"Detected file type: {mime}")

    # prepare inline data
    b64 = file_to_base64(file_path)

    data = {
        "inline_data": {"mime_type": mime, "data": b64}
    }

    # send to model
    result = model.generate_content(
        [EXTRACTION_PROMPT, data]
    )

    return result.text


# ---------------------------------------------------
# TEST
# ---------------------------------------------------
if __name__ == "__main__":
    out = extract_claim_fields("your_input_file.pdf")  # or .jpg, .png
    print("OUTPUT:\n", out)
