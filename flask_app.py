import os
from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('EfficientNetV2S_Leukemia_Fold5.keras')

# Define the folder for uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prepare the image
def prepare_image(img):
    img = img.resize((128, 128))  # Resize to model's expected size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict(img):
    img = prepare_image(img)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    if predicted_class == 1:
        result = "Leukemia"
    else:
        result = "Normal"
    
    return result, confidence


# Upload page
# Prediction route
# Home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)
        
        # Prepare the image and make the prediction
        img = Image.open(img_path)
        result, confidence = predict(img)  # Now getting confidence
        
        return render_template('result.html', result=result, img_path=file.filename, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
