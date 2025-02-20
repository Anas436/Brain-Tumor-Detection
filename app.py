from flask import Flask, request, jsonify,render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.pipeline.prediction import onnxModel_prediction
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
 
    result = onnxModel_prediction(image)
    
    return render_template('index.html',prediction_text=f'{result}')

if __name__ == "__main__":
    app.run(debug=True)
