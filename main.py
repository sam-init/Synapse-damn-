from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import uuid
from ocr.ocr import extract_claim_fields
from agent.agent import run_agent
from infer import load_model, predict

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
    uploaded_files = []

    # Create local upload directory if not exists
    upload_dir = os.path.join(os.getcwd(), "uploads", "claims")
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        save_path = os.path.join(upload_dir, file.filename)
        file.save(save_path)

        # Convert to URL for frontend
        file_url = f"/static_claims/{file.filename}"

        uploaded_files.append({
            "filename": file.filename,
            "path": save_path,
            "url": file_url
        })

    return jsonify({
        "status": "success",
        "uploaded_files": uploaded_files
    })

@app.route("/process-claim", methods=["POST"])
def process_claim():
    """
    Pipeline:
    1. Receive uploaded files
    2. Pick OCR source (PDF/IMG)
    3. Pick damage image
    4. Run OCR
    5. Run CNN
    6. Feed into agent
    """

    files = request.files.getlist("all_files[]")
    if not files:
        return {"error": "No files uploaded"}, 400

    # Save locally
    upload_dir = os.path.join(os.getcwd(), "uploads", "claims")
    os.makedirs(upload_dir, exist_ok=True)

    # Separate OCR file & damage files
    ocr_source = None
    damage_images = []

    saved_paths = []

    for f in files:
        save_path = os.path.join(upload_dir, f.filename)
        f.save(save_path)
        saved_paths.append(save_path)

        if f.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
            if "form" in f.filename.lower() or "claim" in f.filename.lower():
                ocr_source = save_path
            else:
                damage_images.append(save_path)

    if ocr_source is None:
        ocr_source = saved_paths[0]  # fallback

    if len(damage_images) == 0:
        return {"error": "No damage images found"}, 400

    # -------------------------
    # 1. OCR extraction
    # -------------------------
    print("\nRunning OCR on:", ocr_source)
    ocr_output = extract_claim_fields(ocr_source)

    # -------------------------
    # 2. CNN — take first damage image
    # -------------------------
    damage_path = damage_images[0]
    print("\nRunning CNN on:", damage_path)
    damage_label, confidence = predict(damage_path, cnn_model)

    cnn_output = {
        "damage_label": damage_label,
        "confidence": confidence
    }

    # -------------------------
    # 3. Feed into Agent
    # -------------------------
    print("\nSending to Agent…")

    agent_output = run_agent(
        ocr_output,
        cnn_output
    )

    return {
        "ocr": ocr_output,
        "cnn": cnn_output,
        "agent": agent_output
    }

if __name__ == "__main__":
    app.run(debug=True)
