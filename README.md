# hidden-image-detector

i want an automated image steganography decoding web application using this code Creating a web application to detect hidden images using advanced steganography techniques is a complex task. Here’s a comprehensive guide and detailed code to achieve this. We'll use Python's Flask framework for the backend, which will handle the image processing, and HTML/CSS for the frontend. We'll also incorporate machine learning models and image processing libraries for advanced steganalysis.

Prerequisites
Python 3.x
Flask
OpenCV
Pillow
TensorFlow/Keras (for machine learning models)
NumPy
scikit-learn
Step-by-Step Guide
1. Setting Up the Environment
First, install the necessary libraries:

bash
Copy code
pip install Flask opencv-python Pillow tensorflow numpy scikit-learn
2. Create the Flask App Structure
Create a directory structure for the Flask app:

arduino
Copy code
steganalysis_app/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── models/
│   └── steganalysis_model.h5
└── utils/
    └── steganalysis.py
3. Develop the Backend
app.py (Main Flask application):

python
Copy code
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils.steganalysis import detect_steganography

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = detect_steganography(filepath)
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
utils/steganalysis.py (Utility functions for steganalysis):

python
Copy code
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
4. Develop the Frontend
templates/index.html:

html
Copy code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Steganography Detection</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Steganography Detection</h1>
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload and Analyze</button>
        </form>
    </div>
</body>
</html>
templates/result.html:

html
Copy code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Steganography Detection Result</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Analysis Result</h1>
        <p>Is Steganography Detected: {{ result.is_stegano }}</p>
        <p>Hidden Data: {{ result.lsb_data }}</p>
        <a href="/">Go Back</a>
    </div>
</body>
</html>
static/style.css:

css
Copy code
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.container {
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    text-align: center;
}

h1 {
    margin-bottom: 20px;
}

form {
    display: flex;
    flex-direction: column;
    align-items: center;
}

input[type="file"] {
    margin-bottom: 20px;
}

button {
    padding: 10px 20px;
    border: none;
    background-color: #5cb85c;
    color: white;
    cursor: pointer;
}

button:hover {
    background-color: #4cae4c;
}
5. Prepare the Machine Learning Model
You'll need a pre-trained machine learning model for steganalysis. This can be a Convolutional Neural Network (CNN) trained on a dataset of images with and without steganography.

Here is an example of a simple CNN model training script (you need to train it separately and save as stegananalysis_model.h5):

python
Copy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare data (assuming you have datasets of clean and steganographic images)
train_datagen = ImageDataGenerator(rescale=0.2, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'data/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('models/stegananalysis_model.h5')
Running the Application
Start the Flask application by running:

bash
Copy code
python app.py
Open your browser and navigate to http://127.0.0.1:5000/ to use the application.

This detailed setup provides a comprehensive web application for detecting hidden images using advanced steganography techniques.



## Collaborate with GPT Engineer

This is a [gptengineer.app](https://gptengineer.app)-synced repository 🌟🤖

Changes made via gptengineer.app will be committed to this repo.

If you clone this repo and push changes, you will have them reflected in the GPT Engineer UI.

## Tech stack

This project is built with React and Chakra UI.

- Vite
- React
- Chakra UI

## Setup

```sh
git clone https://github.com/GPT-Engineer-App/hidden-image-detector.git
cd hidden-image-detector
npm i
```

```sh
npm run dev
```

This will run a dev server with auto reloading and an instant preview.

## Requirements

- Node.js & npm - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
