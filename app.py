from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

MODEL_DIR = "./plant_model_local"

print("Loading model from local folder...")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
print("Model loaded instantly! Server ready.")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(-1)[0]
    
    confidence, idx = torch.max(probs, 0)
    disease = model.config.id2label[idx.item()]
    
    return jsonify({
        "crop": disease.split("___ ")[0],
        "disease": disease.replace("___", " - "),
        "confidence": f"{confidence.item():.1%}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)