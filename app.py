from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your pre-trained model (replace 'model.pkl' with your model's filename)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json

    # Convert the data into a NumPy array for prediction
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction using the model
    prediction = model.predict(features)

    # Return the result as JSON
    return jsonify({
        'prediction': prediction[0]
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
