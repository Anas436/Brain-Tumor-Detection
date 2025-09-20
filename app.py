from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src/pipeline")))
from src.pipeline.prediction import onnxModel_prediction  # make sure the import matches your folder structure

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        result = onnxModel_prediction(image)
        # Return JSON for AJAX to handle
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

