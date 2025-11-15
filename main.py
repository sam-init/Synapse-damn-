from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import uuid
from ocr.ocr import extract_claim_fields
from agent.agent import run_agent
from infer import load_model, predict

cnn_model = load_model()

# Serve uploaded images at /static_claims/<file>
app = Flask(
    __name__,
    static_folder='uploads/claims'
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


# -----------------------------
# Upload Files (raw)
# -----------------------------
@app.route("/upload-files", methods=["POST"])
def upload_files():
    if "all_files[]" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("all_files[]")
    uploaded_files = []

    upload_dir = os.path.join("uploads", "claims")
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        save_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(save_path)

        uploaded_files.append({
            "filename": f.filename,
            "path": save_path.replace("\\", "/"),
            "url": "/static_claims/" + f.filename
        })

    return jsonify({
        "status": "success",
        "uploaded_files": uploaded_files
    })


# -----------------------------
# Full pipeline: OCR + CNN + Agent
# -----------------------------
@app.route("/process-claim", methods=["POST"])
def process_claim():

    files = request.files.getlist("all_files[]")
    if not files:
        return {"error": "No files uploaded"}, 400

    upload_dir = os.path.join("uploads", "claims")
    os.makedirs(upload_dir, exist_ok=True)

    pdf_file = None
    image_files = []
    file_paths = []

    # Separate PDF and images
    for f in files:
        save_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(save_path)
        file_paths.append(save_path)

        if f.filename.lower().endswith(".pdf"):
            pdf_file = save_path
        else:
            image_files.append(save_path)

    if pdf_file is None:
        return {"error": "No PDF uploaded"}, 400

    if not image_files:
        return {"error": "No image files"}, 400

    # OCR
    print("Running OCR on:", pdf_file)
    ocr_output = extract_claim_fields(pdf_file)

    # CNN
    cnn_outputs = []
    for img in image_files:
        label, conf = predict(img, cnn_model)
        cnn_outputs.append({
            "file": img,
            "label": label,
            "confidence": conf
        })

    strongest = max(cnn_outputs, key=lambda x: x["confidence"])

    cnn_final = {
        "damage_label": strongest["label"],
        "confidence": strongest["confidence"],
        "all_results": cnn_outputs
    }

    # Agent
    agent_output = run_agent(
        form_data=ocr_output,
        image_data=cnn_final,
        question=None
    )

    # Save JSON for admin
    import json
    claim_id = f"CLAIM-{uuid.uuid4().hex[:6]}"
    json_path = os.path.join(upload_dir, f"{claim_id}.json")

    with open(json_path, "w") as f:
        json.dump({
            "claim_id": claim_id,
            "ocr": ocr_output,
            "cnn": cnn_final,
            "agent": agent_output,
            "uploaded_files": file_paths
        }, f, indent=2)

    # FIXED: return URLs properly
    uploaded_files_data = [
        {
            "path": p.replace("\\", "/"),
            "url": "/static_claims/" + os.path.basename(p)
        }
        for p in file_paths
    ]

    return {
        "claim_id": claim_id,
        "ocr": ocr_output,
        "cnn": cnn_final,
        "agent": agent_output,
        "uploaded_files": uploaded_files_data
    }


@app.route("/ask-agent", methods=["POST"])
def ask_agent():
    data = request.json
    reply = run_agent(
        form_data=data.get("form_data"),
        image_data=data.get("image_data"),
        question=data.get("question")
    )
    return jsonify({"reply": reply})


@app.route("/result")
def result_page():
    return render_template("result.html")


# -----------------------------
# Admin: load all claims
# -----------------------------
@app.route("/admin")
def admin_page():
    claims = []
    cdir = os.path.join("uploads", "claims")

    for f in os.listdir(cdir):
        if f.endswith(".json"):
            import json
            with open(os.path.join(cdir, f), "r") as jf:
                claims.append(json.load(jf))

    return render_template("admin.html", claims=claims)


@app.route("/admin/action", methods=["POST"])
def admin_action():
    data = request.json
    claim_id = data["claim_id"]
    action = data["action"]

    json_path = os.path.join("uploads", "claims", f"{claim_id}.json")

    if not os.path.exists(json_path):
        return {"error": "Claim not found"}, 404

    import json
    with open(json_path, "r") as f:
        claim = json.load(f)

    claim["admin_action"] = action

    with open(json_path, "w") as f:
        json.dump(claim, f, indent=2)

    return {"status": "success"}


if __name__ == "__main__":
    app.run(debug=True)
