import joblib
import numpy as np
import pandas as pd

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Recommends the best crop based on soil nutrients and climate conditions.
    
    Parameters:
    -----------
    N : int or float
        Nitrogen content in soil (kg/ha)
    P : int or float
        Phosphorus content in soil (kg/ha)
    K : int or float
        Potassium content in soil (kg/ha)
    temperature : float
        Temperature in degrees Celsius
    humidity : float
        Relative humidity in percentage
    ph : float
        pH value of the soil
    rainfall : float
        Rainfall in mm
        
    Returns:
    --------
    str
        The name of the recommended crop
    """
    # Load the trained model
    try:
        model = joblib.load('crop_recommendation_model.pkl')
    except FileNotFoundError:
        return "Error: Model file 'crop_recommendation_model.pkl' not found. Make sure to train the model first."
    
    # Create input array for prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Make prediction
    crop = model.predict(input_data)[0]
    
    return crop

def get_top_recommendations(N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    """
    Returns the top N recommended crops with their probability scores.
    
    Parameters:
    -----------
    N, P, K, temperature, humidity, ph, rainfall : Same as recommend_crop function
    top_n : int, optional (default=3)
        Number of top recommendations to return
        
    Returns:
    --------
    list of tuples
        List of (crop, probability) tuples sorted by probability in descending order
    """
    # Load the trained model
    try:
        model = joblib.load('crop_recommendation_model.pkl')
    except FileNotFoundError:
        return [("Error: Model file not found", 0.0)]
    
    # Create input array for prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Get probability scores for all classes
    probabilities = model.predict_proba(input_data)[0]
    
    # Sort indices by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    
    # Get crop names and their corresponding classes
    class_labels = model.classes_
    
    # Create list of (crop, probability) tuples
    recommendations = []
    for i in range(min(top_n, len(class_labels))):
        idx = sorted_indices[i]
        crop = class_labels[idx]
        probability = probabilities[idx]
        recommendations.append((crop, probability))
    
    return recommendations

# Example usage
if __name__ == "__main__":
    # You can either take input from user or hardcode values for testing
    
    # Method 1: Hardcoded values for testing
    print("\nExample 1: Using hardcoded values")
    N, P, K = 90, 42, 43
    temperature = 20.9
    humidity = 82.0
    ph = 6.5
    rainfall = 202.9
    
    # Get single best recommendation
    recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
    print(f"Recommended crop: {recommended_crop}")
    
    # Get top 3 recommendations with probabilities
    top_recommendations = get_top_recommendations(N, P, K, temperature, humidity, ph, rainfall)
    print("Top recommendations:")
    for i, (crop, probability) in enumerate(top_recommendations, 1):
        print(f"{i}. {crop} - {probability:.4f} probability")
    
    # Method 2: Taking input from user
    print("\nExample 2: Taking input from user")
    try:
        N = float(input("Enter Nitrogen content (N): "))
        P = float(input("Enter Phosphorus content (P): "))
        K = float(input("Enter Potassium content (K): "))
        temperature = float(input("Enter temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        ph = float(input("Enter pH value: "))
        rainfall = float(input("Enter rainfall (mm): "))
        
        # Get single best recommendation
        recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
        print(f"\nRecommended crop: {recommended_crop}")
        
        # Get top 3 recommendations with probabilities
        top_recommendations = get_top_recommendations(N, P, K, temperature, humidity, ph, rainfall)
        print("\nTop recommendations:")
        for i, (crop, probability) in enumerate(top_recommendations, 1):
            print(f"{i}. {crop} - {probability:.4f} probability")
    except ValueError:
        print("Error: Please enter numeric values for all inputs.")