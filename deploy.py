from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from train import iou  # Import your custom metric if needed
import base64

app = Flask(__name__)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def predict_single_image(model, image_path):
    # Read and preprocess the image
    image = read_image(image_path)
    
    # Predict segmentation mask
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    
    # Convert prediction to binary mask (thresholding)
    binary_mask = (prediction > 0.5).astype(np.uint8)
    
    return binary_mask

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Load the trained model
        with CustomObjectScope({'iou': iou}):  # Use if custom metric is used
            model = tf.keras.models.load_model("files/model.h5")
        
        # Get the uploaded image file
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join("uploads", image_file.filename)
            image_file.save(image_path)
            
            # Predict segmentation mask for the single image
            segmented_image = predict_single_image(model, image_path)
            
            # Convert segmented image to base64 encoded string
            _, buffer = cv2.imencode('.jpg', segmented_image * 255)
            segmented_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return render_template('result.html', segmented_image=segmented_image_base64)

    return render_template('index.html')
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        # Load the trained model
        with CustomObjectScope({'iou': iou}):  # Use if custom metric is used
            model = tf.keras.models.load_model("files/model.h5")
        
        # Get the uploaded image file
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join("uploads", image_file.filename)
            image_file.save(image_path)
            
            # Read and preprocess the uploaded image
            uploaded_image = read_image(image_path)
            
            # Predict segmentation mask for the uploaded image
            segmented_image = predict_single_image(model, image_path)
            
            # Convert images to base64 encoded strings
            _, buffer = cv2.imencode('.jpg', uploaded_image * 255)
            uploaded_image_base64 = base64.b64encode(buffer).decode('utf-8')
            _, buffer = cv2.imencode('.jpg', segmented_image * 255)
            segmented_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return render_template('result.html', uploaded_image=uploaded_image_base64, segmented_image=segmented_image_base64)

    return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
