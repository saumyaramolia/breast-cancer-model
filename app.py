import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))


# Define an API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data as JSON
        data = request.get_json()

        # Ensure that the data contains all the required features
        required_features = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Prepare the input data as a numpy array
        features = np.array([data[feature] for feature in required_features]).reshape(1, -1)

        # Make a prediction using the model
        prediction = model.predict(features)

        # Return the prediction as JSON
        result = {"diagnosis": "Malignant" if prediction[0] == 1 else "Benign"}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
