import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained machine learning model
model = load_model('models/steganalysis_model.h5')

def extract_lsb(image):
    binary_data = ""
    pixels = np.array(image)
    for pixel in pixels.flatten():
        binary_data += bin(pixel)[-1]
    byte_data = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    decoded_data = ''.join([chr(int(byte, 2)) for byte in byte_data if int(byte, 2) != 0])
    return decoded_data

def detect_steganography(image_path):
    image = Image.open(image_path).convert('RGB')
    lsb_data = extract_lsb(image)
    
    # Convert image to numpy array and preprocess for the model
    image_array = np.array(image.resize((128, 128))) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict using the pre-trained model
    prediction = model.predict(image_array)
    is_stegano = prediction[0][0] > 0.5
    
    result = {
        'is_stegano': is_stegano,
        'lsb_data': lsb_data if is_stegano else "No hidden data found"
    }
    return result