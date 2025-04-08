# Simple NPK Prediction Input Tool
import pandas as pd
import joblib

def get_npk_prediction():
    # Load the saved model
    try:
        model_data = joblib.load('npk_prediction_model.pkl')
        models = model_data['models']
    except:
        print("Error: Could not load model. Make sure 'npk_prediction_model.pkl' exists.")
        return
    
    # Available crops list
    available_crops = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
        'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
    ]
    
    # Print available crops
    print("\nAvailable crops:")
    for i, crop in enumerate(available_crops, 1):
        print(f"{i}. {crop}")
    
    # Get user input
    try:
        crop_index = int(input("\nEnter crop number from the list above: ")) - 1
        if crop_index < 0 or crop_index >= len(available_crops):
            print("Invalid crop number.")
            return
        
        crop = available_crops[crop_index]
        temperature = float(input("Enter temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        ph = float(input("Enter pH value: "))
        rainfall = float(input("Enter rainfall (mm): "))
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'label': [crop],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        
        # Make predictions
        n_pred = models['N'].predict(input_data)[0]
        p_pred = models['P'].predict(input_data)[0]
        k_pred = models['K'].predict(input_data)[0]
        
        # Print results
        print("\n" + "="*50)
        print(f"PREDICTION RESULTS FOR: {crop.upper()}")
        print("="*50)
        print(f"Environmental Conditions:")
        print(f"  Temperature: {temperature}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  pH: {ph}")
        print(f"  Rainfall: {rainfall} mm")
        print("\nPredicted NPK Values:")
        print(f"  N (Nitrogen): {round(n_pred)}")
        print(f"  P (Phosphorus): {round(p_pred)}")
        print(f"  K (Potassium): {round(k_pred)}")
        print("="*50)
        
    except ValueError:
        print("Error: Please enter valid numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("NPK PREDICTION TOOL")
    print("This tool predicts the N, P, K values needed for a crop based on environmental conditions.")
    
    while True:
        get_npk_prediction()
        
        again = input("\nMake another prediction? (y/n): ").lower()
        if again != 'y':
            break
    
    print("Thank you for using the NPK Prediction Tool!")