# app.py

import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "ML Model is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']  # Expects a list of 4 numbers
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
