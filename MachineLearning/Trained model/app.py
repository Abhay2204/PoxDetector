from flask import Flask, request, render_template, jsonify
import tensorflow.lite as tflite
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the TFLite model
def load_model():
    interpreter = tflite.Interpreter(model_path="chickenpox_model_improved.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ["Chickenpox", "Measles", "Monkeypox", "Normal"]

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Ensure 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    image_path = os.path.join('static', file.filename)
    file.save(image_path)
    
    # Preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Invalid image file'})

    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    result = class_labels[prediction]
    
    return jsonify({'prediction': result, 'image_url': f'/{image_path}'})


if __name__ == '__main__':
    app.run(debug=True)
