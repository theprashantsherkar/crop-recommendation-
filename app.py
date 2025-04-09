from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models
def load_crop_model():
    try:
        return joblib.load('crop_recommendation_model.pkl')
    except FileNotFoundError:
        return None

def load_npk_model():
    try:
        return joblib.load('npk_prediction_model.pkl')
    except FileNotFoundError:
        return None

# Helper function for crop recommendations
def get_top_crop_recommendations(N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    """Returns the top N recommended crops with their probability scores."""
    model = load_crop_model()
    if model is None:
        return [{"error": "Crop model file not found"}]
    
    # Create input array for prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Get probability scores for all classes
    probabilities = model.predict_proba(input_data)[0]
    
    # Sort indices by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    
    # Get crop names and their corresponding classes
    class_labels = model.classes_
    
    # Create list of recommendations
    recommendations = []
    for i in range(min(top_n, len(class_labels))):
        idx = sorted_indices[i]
        crop = class_labels[idx]
        probability = float(probabilities[idx])  # Convert to float for JSON serialization
        recommendations.append({
            "crop": crop,
            "probability": probability
        })
    
    return recommendations

# Helper function for NPK predictions
def predict_npk(crop, temperature, humidity, ph, rainfall):
    """Predicts the NPK values for a given crop and environmental conditions."""
    model_data = load_npk_model()
    if model_data is None:
        return {"error": "NPK model file not found"}
    
    models = model_data['models']
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'label': [crop],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    
    try:
        # Make predictions
        n_pred = models['N'].predict(input_data)[0]
        p_pred = models['P'].predict(input_data)[0]
        k_pred = models['K'].predict(input_data)[0]
        
        # Round to integers
        return {
            'N': int(round(n_pred)),
            'P': int(round(p_pred)),
            'K': int(round(k_pred))
        }
    except Exception as e:
        return {"error": str(e)}

# Route for home page
@app.route('/')
def home():
    return jsonify({
        "message": "Crop Recommendation API",
        "endpoints": {
            "/api/recommend-crop": "POST - Recommend crops based on NPK values and environmental conditions",
            "/api/recommend-npk": "POST - Recommend NPK values based on crop type and environmental conditions",
            "/api/crop-list": "GET - Get list of available crops"
        }
    })

# Route for crop recommendations
@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    
    try:
        # Extract values from request
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))
        top_n = int(data.get('top_n', 3))  # Default to top 3 recommendations
        
        # Get recommendations
        recommendations = get_top_crop_recommendations(
            N, P, K, temperature, humidity, ph, rainfall, top_n
        )
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# Route for NPK recommendations
@app.route('/api/recommend-npk', methods=['POST'])
def recommend_npk():
    data = request.get_json()
    
    try:
        # Extract values from request
        crop = data.get('crop')
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))
        
        # Get NPK prediction
        npk_prediction = predict_npk(
            crop, temperature, humidity, ph, rainfall
        )
        
        if "error" in npk_prediction:
            return jsonify({
                "success": False,
                "error": npk_prediction["error"]
            }), 400
        
        return jsonify({
            "success": True,
            "npk_prediction": npk_prediction
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# Route to get list of available crops
@app.route('/api/crop-list', methods=['GET'])
def get_crops():
    available_crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
        'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
    ]
    
    return jsonify({
        "success": True,
        "crops": available_crops
    })

if __name__ == '__main__':
    app.run(debug=True)