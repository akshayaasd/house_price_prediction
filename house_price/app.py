from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the model
model = joblib.load('house_price_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    features = [
        float(request.form['bedrooms']),
        float(request.form['bathrooms']),
        float(request.form['sqft_living']),
        float(request.form['sqft_lot']),
        float(request.form['floors']),
        float(request.form['waterfront']),
        float(request.form['view']),
        float(request.form['condition']),
        float(request.form['grade']),
        float(request.form['sqft_above']),
        float(request.form['sqft_basement']),
        float(request.form['yr_built']),
        float(request.form['yr_renovated']),
        float(request.form['zipcode']),
        float(request.form['lat']),
        float(request.form['long']),
        float(request.form['sqft_living15']),
        float(request.form['sqft_lot15'])
    ]

    # Make prediction
    prediction = model.predict([features])
    return jsonify({'price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
