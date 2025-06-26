
# Create your views here.
import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings # For accessing BASE_DIR

# Define the path to the trained model
# It's good practice to place your model file in a dedicated directory,
# e.g., 'crop_recommender_project/models/best_crop_recommender_model.joblib'
# For simplicity, we'll assume it's in the project root for now.
# Make sure 'best_crop_recommender_model.joblib' is copied to the
# 'crop_recommender_project' (root project) directory.
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model/best_crop_recommender_model.joblib')

# Load the model globally when the server starts to avoid reloading on each request
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}. Make sure you've run train_model.py and copied the .joblib file.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the model: {e}")
    
def index(request):
    return render(request, 'index.html')

def predict_crop(request):
    """
    Handles the crop recommendation prediction.
    GET: Displays the input form.
    POST: Processes input, makes prediction, and displays result.
    """
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        if model is None:
            error_message = "Prediction model is not loaded. Please check the server logs."
        else:
            try:
                # Get input values from the form
                # Ensure these match the features used during training and their order
                N = float(request.POST.get('nitrogen'))
                P = float(request.POST.get('phosphorus'))
                K = float(request.POST.get('potassium'))
                temperature = float(request.POST.get('temperature'))
                humidity = float(request.POST.get('humidity'))
                ph = float(request.POST.get('ph'))
                rainfall = float(request.POST.get('rainfall'))

                # Create a DataFrame for prediction.
                # The column names and order MUST match the training data.
                input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

                # Make prediction
                prediction_result = model.predict(input_data)[0]
                print(f"Prediction made: {prediction_result}")

            except ValueError:
                error_message = "Invalid input. Please ensure all fields are numbers."
            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"

    # Render the template with the prediction result or error message
    return render(request, 'predict.html', {
        'prediction_result': prediction_result,
        'error_message': error_message
    })
    

