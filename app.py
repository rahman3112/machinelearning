import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import logging
import traceback
import os

# Set up logging
logging.basicConfig(level=logging.INFO)#This sets the minimum logging level to INFO only messages with a severity level of INFO or higher (like WARNING, ERROR, and CRITICAL)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = 'HighlyAccurateItemRatingPredictor.pkl'
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    else:
        logger.error(f"Model file not found: {model_path}")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        data = request.json

        # Extract data from the request
        day = int(data['day'])
        month = int(data['month'])
        year = int(data['year'])
        cloth_type = data['cloth_type']
        payment_method = int(data['payment_method'])
        purchase_amount = float(data['purchase_amount'])
        actual_rating = data.get('actual_rating')  # This would be provided if available

        # Calculate day of week and is_weekend
        date = pd.to_datetime(f"{year}-{month}-{day}")
        day_of_week = date.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0

        # Map cloth_type to its index
        cloth_types = ['Belt', 'Skirt', 'T-shirt', 'Jeans', 'Sneakers', 'Dress', 'Handbag', 'Jacket', 'Sweater', 'Scarf']
        cloth_type_index = cloth_types.index(cloth_type)

        # Prepare input data for the model
        input_data = pd.DataFrame([{
            'Month': month,
            'DayOfWeek': day_of_week,
            'IsWeekend': is_weekend,
            'PaymentMethod': 'Credit Card' if payment_method == 1 else 'Cash',
            'PurchaseAmount': purchase_amount,
            'Item': cloth_type
        }])

        # Make prediction
        prediction = model.predict(input_data)

        # Adjust prediction to 1-5 scale
        adjusted_prediction = int(prediction[0])

        # Log prediction details
        log_message = f"Prediction request: Date={date.date()}, Item={cloth_type}, " \
                      f"Payment={payment_method}, Amount=${purchase_amount:.2f}, " \
                      f"Predicted Rating={adjusted_prediction}"

        # Calculate and log accuracy if actual rating is provided
        if actual_rating is not None:
            accuracy = 1 if adjusted_prediction == actual_rating else 0
            log_message += f", Actual Rating={actual_rating}, Accuracy={accuracy}"
        
        logger.info(log_message)

        return jsonify({'rating': adjusted_prediction})

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port)
