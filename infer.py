import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "weights/vehicle_damage_resnet18.pth"
CLASSES = ["minor", "moderate", "severe"]  # must match training order

# FORCE CUDA (you said "dont use cpu")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available. Fix your CUDA/driver setup first.")

DEVICE = torch.device("cuda")
print("Using device:", DEVICE, f"({torch.cuda.get_device_name(0)})")

# -----------------------
# TRANSFORMS (same as val/test)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------
# LOAD MODEL
# -----------------------
def load_model():
    # Same base architecture as in train.py
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASSES))

    # Load checkpoint
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # --- IMPORTANT PART ---
    # If the checkpoint was saved when model.fc was Sequential(Dropout, Linear),
    # the keys will be "fc.1.weight"/"fc.1.bias".
    # We remap them to "fc.weight"/"fc.bias" so this Linear head can load them.
    if "fc.1.weight" in state_dict and "fc.weight" not in state_dict:
        state_dict["fc.weight"] = state_dict["fc.1.weight"]
        state_dict["fc.bias"] = state_dict["fc.1.bias"]
        del state_dict["fc.1.weight"]
        del state_dict["fc.1.bias"]

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict(img_path, model):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    return CLASSES[idx.item()], conf.item()

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    model = load_model()

    img_path = input("Enter image path: ").strip()

    # Allow simple names like "images.jpg" from your project root
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.getcwd(), img_path)

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")

    print("Using image:", img_path)

    label, confidence = predict(img_path, model)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}")
