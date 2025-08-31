from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from PIL import Image
import io

# Load the trained model (using joblib for .pkl)
model = joblib.load('model.pkl')  # Adjust this path if necessary

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Load and preprocess image
        img = Image.open(io.BytesIO(file.read()))  # Open the image from the file stream
        img = img.resize((256, 256))  # Resize the image to the required size
        img_array = np.array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img_array)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})  # Assuming a list output from the model

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
