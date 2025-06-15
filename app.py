from flask import Flask, request, render_template, jsonify
import joblib
from PIL import Image
import numpy as np
import os
from tensorflow.keras.utils import img_to_array
app = Flask(__name__)

# Load your model
model = joblib.load("model/disease_detection_model.pkl")

label_map = {0:"COVID-19", 1:"Normal", 2:"Viral Pneumonia", 3:"Bacterial Pneumonia"}

def preprocess_image(image):
    # Resize image to expected input size (224x224), convert to array, and normalize
    image = image.resize((256, 256))  # Resize to match model input
    image = image.convert("RGB")  # Ensure 3 channels
    image = img_to_array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize to 0-1
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        if prediction.shape[1] == 1:
            # Binary classification (e.g., 0 = Normal, 1 = Diseased)
            predicted_class = int(prediction[0][0] > 0.5)
            confidence = float(prediction[0][0])
        else:
            # Multi-class classification
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction))

        return jsonify({
            "prediction": label_map[int(predicted_class)],
            "confidence": round(confidence, 4)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
