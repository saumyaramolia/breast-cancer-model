from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
CORS(app, origins='*')

# Load the trained model
model = pickle.load(open("./model/model.pkl", "rb"))

# Define the selected features
selected_features = ['area_mean', 'area_se', 'concavity_mean', 'concavity_se', 'concavity_worst',
                     'fractal_dimension_se',
                     'fractal_dimension_worst', 'smoothness_worst', 'symmetry_worst', 'texture_mean']


# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [data[feature] for feature in selected_features]
        prediction = model.predict([features])

        # Convert the prediction to a human-readable label
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        return jsonify({"prediction": diagnosis})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main':
    app.run()
