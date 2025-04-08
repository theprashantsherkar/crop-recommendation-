import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nUnique Crops:", df['label'].unique())
print("Number of Unique Crops:", df['label'].nunique())

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Initialize and train the Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=15,      # Maximum depth of trees
    min_samples_split=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save the model
joblib.dump(rf_model, 'crop_recommendation_model.pkl')
print("\nModel saved as 'crop_recommendation_model.pkl'")

# Function to recommend crops based on input values
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    
    # Make prediction
    prediction = rf_model.predict(input_data)[0]
    
    # Get probability scores for all classes
    probabilities = rf_model.predict_proba(input_data)[0]
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
    
    # Get top 3 predictions with probabilities
    top_predictions = []
    class_labels = rf_model.classes_
    
    for i in range(3):  # Get top 3
        idx = sorted_indices[i]
        crop = class_labels[idx]
        probability = probabilities[idx]
        top_predictions.append((crop, probability))
    
    return top_predictions

# Example usage
print("\nExample Recommendations:")

test_cases = [
    {'N': 90, 'P': 42, 'K': 43, 'temperature': 21, 'humidity': 82, 'ph': 6.5, 'rainfall': 200},
    {'N': 20, 'P': 100, 'K': 90, 'temperature': 25, 'humidity': 70, 'ph': 6.8, 'rainfall': 150},
    {'N': 120, 'P': 30, 'K': 30, 'temperature': 22, 'humidity': 65, 'ph': 7.0, 'rainfall': 100},
    {'N': 60, 'P': 60, 'K': 60, 'temperature': 28, 'humidity': 40, 'ph': 8.5, 'rainfall': 40}
]

for i, case in enumerate(test_cases, 1):
    results = recommend_crop(**case)
    print(f"\nCase {i}:")
    print(f"Input: {case}")
    print("Top recommendations:")
    for crop, prob in results:
        print(f"- {crop}: {prob:.4f} probability")