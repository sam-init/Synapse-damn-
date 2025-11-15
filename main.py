from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import uuid
from utils import upload_to_supabase
from ocr.ocr import extract_claim_fields
from agent.agent import run_agent

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------ INDEX PAGE ------------------
@app.route("/")
def index():
    return render_template("index.html")


# ------------------ OCR-ONLY ROUTE ------------------
@app.route("/ocr-extract", methods=["POST"])
def ocr_extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        ocr_result = extract_claim_fields(save_path)
        return jsonify({"ocr_output": ocr_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# -------------- FULL PIPELINE: PROCESS CLAIM --------------
@app.route("/process-claim", methods=["POST"])
def process_claim():
    form_file = request.files.get("form")
    damage_files = request.files.getlist("damage")

    if not form_file:
        return jsonify({"error": "Claim form missing"}), 400

    # Save form file
    form_filename = secure_filename(f"{uuid.uuid4()}_{form_file.filename}")
    form_path = os.path.join(UPLOAD_FOLDER, form_filename)
    form_file.save(form_path)

    # Save damage images
    damage_paths = []
    for d in damage_files:
        filename = secure_filename(f"{uuid.uuid4()}_{d.filename}")
        path = os.path.join(UPLOAD_FOLDER, filename)
        d.save(path)
        damage_paths.append(path)

    # OCR
    form_data = extract_claim_fields(form_path)

    # CNN placeholder
    image_data = {
        "damage_location": "PENDING",
        "damage_severity": "PENDING"
    }

    # AI Agent
    agent_reply = run_agent(form_data, image_data, question=None)

    return jsonify({
        "form_data": form_data,
        "image_analysis": image_data,
        "agent_reply": agent_reply
    })

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

@app.route("/upload-files", methods=["POST"])
def upload_files():
    if "all_files[]" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("all_files[]")
    uploaded = []

    temp_dir = "uploads"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        save_path = os.path.join(temp_dir, file.filename)
        file.save(save_path)

        url = upload_to_supabase(save_path)

        uploaded.append({
            "filename": file.filename,
            "url": url
        })

    return jsonify({
        "status": "success",
        "uploaded_files": uploaded
    })


if __name__ == "__main__":
    app.run(debug=True)
