
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler and model
with open('wine_quality_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('wine_quality_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    input_data = request.json
    try:
        features = np.array([input_data['features']]).astype(float)
        # Scale the features
        scaled_features = scaler.transform(features)
        # Predict using the model
        prediction = model.predict(scaled_features)
        # Return the prediction
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
