import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from train import iou  

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

if __name__ == "__main__":
    # Load the trained model
    with CustomObjectScope({'iou': iou}):  # Use if custom metric is used
        model = tf.keras.models.load_model("files/model.h5")
    
    # Path to the single image you want to predict
    image_path = "/home//bobby//Desktop//poly----//New folder//1.tif"
    
    # Predict segmentation mask for the single image
    segmented_image = predict_single_image(model, image_path)
    
    # Save the segmented image in results_pred folder
    save_folder = "results_pred"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "segmented_image.jpg")
    cv2.imwrite(save_path, segmented_image * 255)
    
    print(f"Segmented image saved at: {save_path}")
