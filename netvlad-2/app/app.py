from flask import Flask, request, jsonify
from utils import extract_descriptor
import torch
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

@app.route("/compare", methods=["POST"])
def compare():
    file1 = request.files.get("image1")
    file2 = request.files.get("image2")

    if not file1 or not file2:
        return jsonify({"error": "Missing files"}), 400

    desc1 = extract_descriptor(file1.read())
    desc2 = extract_descriptor(file2.read())
    sim = cosine_similarity(desc1, desc2)

    return jsonify({
        "similarity": sim,
        "result": "same" if sim > 0.75 else "different"
    })

if __name__ == "__main__":
    app.run(debug=True)
