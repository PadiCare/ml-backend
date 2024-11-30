import json
from flask import Flask, request, jsonify
from google.cloud import storage
import numpy as np
from google.oauth2 import service_account
import os
import tensorflow as tf
import tempfile

# Set environment variable within the script
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceaccount-ml/secret-ml'

# Load service account credentials
credentials = service_account.Credentials.from_service_account_file(
    'serviceaccount-ml/secret-ml'  # Path to your service account key file
)

app = Flask(__name__)

# Initialize the Google Cloud Storage client with the credentials
storage_client = storage.Client(credentials=credentials)

# Initialize Google Cloud Storage bucket
with open("config.json") as config_file:
    config = json.load(config_file)
bucket_model = storage_client.bucket(config['bucketModel'])
bucket_images = storage_client.bucket(config['bucketPaddy'])

# URL ke model di Google Cloud Storage
model_url = 'gs://paddymodel/model_padicare.h5'
model = None  # Initialize model as None

# List of class labels (modify this to match your model's classes)
class_labels = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

# Function to download the model from Google Cloud Storage
def load_model_from_gcs(model_url):
    global model
    if model is None:
        blob = bucket_model.blob(model_url.split('/')[-1])  # Extract blob name from URL
        
        # Create a temporary file to store the model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_model_file:
            blob.download_to_filename(temp_model_file.name)  # Save model to the temporary file
            model = tf.keras.models.load_model(temp_model_file.name)  # Load model from file
        
        print("Model loaded from GCS")
    return model

# Function to get image from Cloud Storage
def get_image_from_storage(image_id):
    blob = bucket_images.blob(image_id)
    if not blob.exists():
        raise Exception("Image not found in storage")
    image_data = blob.download_as_bytes()
    return image_data

# Function to perform prediction
def predict_image(image_data):
    image = tf.io.decode_image(image_data, channels=3)
    image = tf.image.resize(image, [224, 224])  # Adjust size according to your model
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    label_index = np.argmax(predictions, axis=1)[0]  # Get label index with argmax
    label = class_labels[label_index]  # Map index to class label
    confidence = float(predictions[0][label_index])  # Get confidence score
    return label, confidence  # Return label and confidence score

# Route for the root URL
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the ML Prediction Service!"})

# Endpoint to receive prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_id = data['imageId']
    print(f"Received imageId: {image_id}")
    try:
        load_model_from_gcs(model_url)  # Load model when a prediction is requested
        image_data = get_image_from_storage(image_id)  # Get image from Cloud Storage
        prediction_label = predict_image(image_data)  # Perform prediction
        
        # Print the prediction result
        print(f"Prediction result for imageId {image_id} are {prediction_label}")
        
        return jsonify({'label': prediction_label})  # Return prediction label
    except Exception as e:
        print('Prediction error:', e)
        return jsonify({'error': 'Failed to predict image'}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Gunakan port dari environment
    app.run(host='0.0.0.0', port=port)