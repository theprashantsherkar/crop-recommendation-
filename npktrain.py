# NPK Prediction Model based on Crop type and environmental factors
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Define features and targets
X = df[['label', 'temperature', 'humidity', 'ph', 'rainfall']]
y_n = df['N']
y_p = df['P']
y_k = df['K']

# Create preprocessor for categorical data (crop type)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['label'])
    ],
    remainder='passthrough'
)

# Split the data into training and testing sets
X_train, X_test, y_n_train, y_n_test, y_p_train, y_p_test, y_k_train, y_k_test = train_test_split(
    X, y_n, y_p, y_k, test_size=0.2, random_state=42
)

# Create pipelines for each target (N, P, K)
n_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

p_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

k_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the models
print("Training N model...")
n_pipeline.fit(X_train, y_n_train)

print("Training P model...")
p_pipeline.fit(X_train, y_p_train)

print("Training K model...")
k_pipeline.fit(X_train, y_k_train)

# Evaluate the models
n_pred = n_pipeline.predict(X_test)
p_pred = p_pipeline.predict(X_test)
k_pred = k_pipeline.predict(X_test)

print("\nModel Performance:")
print(f"N - R² Score: {r2_score(y_n_test, n_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_n_test, n_pred)):.4f}")
print(f"P - R² Score: {r2_score(y_p_test, p_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_p_test, p_pred)):.4f}")
print(f"K - R² Score: {r2_score(y_k_test, k_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_k_test, k_pred)):.4f}")

# Create models dictionary
models = {
    'N': n_pipeline,
    'P': p_pipeline,
    'K': k_pipeline
}

# Save the model
print("\nSaving model...")
model_data = {
    'models': models,
    'preprocessor': preprocessor
}
joblib.dump(model_data, 'npk_prediction_model.pkl')
print("Model saved to npk_prediction_model.pkl")

# Function to predict NPK for a single input
def predict_npk(crop, temperature, humidity, ph, rainfall):
    # Load the model
    model_data = joblib.load('npk_prediction_model.pkl')
    models = model_data['models']
    
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
    
    # Round to integers
    return {
        'N': round(n_pred),
        'P': round(p_pred),
        'K': round(k_pred)
    }

# Example usage
if __name__ == "__main__":
    # Example prediction
    crop = 'rice'
    temperature = 24.0
    humidity = 80.0
    ph = 6.5
    rainfall = 200.0
    
    print("\nMaking prediction...")
    npk = predict_npk(crop, temperature, humidity, ph, rainfall)
    print(f"Predicted N, P, K for {crop}: N={npk['N']}, P={npk['P']}, K={npk['K']}")