from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all domains

# Load dummy ResNet18 model for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.models import resnet18
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).squeeze()
    return features

def cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

@app.route("/compare", methods=["POST"])
def compare_images():
    file1 = request.files.get("image1")
    file2 = request.files.get("image2")

    if not file1 or not file2:
        return jsonify({"error": "Missing files"}), 400

    feat1 = extract_features(file1.read())
    feat2 = extract_features(file2.read())

    similarity = cosine_similarity(feat1, feat2)

    return jsonify({
        "similarity": similarity,
        "result": "same" if similarity > 0.85 else "different"
    })

if __name__ == "__main__":
    app.run(debug=True)
