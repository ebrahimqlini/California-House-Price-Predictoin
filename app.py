import pickle
from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and the scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Receive data from the request
    data = request.json['data']
    
    # Convert data to a DataFrame to maintain feature names and avoid UserWarnings
    # The order depends on the keys provided in the JSON input
    df_input = pd.DataFrame([list(data.values())], columns=list(data.keys()))
    
    # Scale the input data using the loaded scaler
    new_data = scalar.transform(df_input)
    
    # Perform prediction using the loaded regression model
    output = regmodel.predict(new_data)
    
    # Round the final result to 4 decimal places
    final_output = round(float(output[0]), 4)
    
    # Print the result to the terminal for debugging purposes
    print(f"Prediction: {final_output}")
    
    return jsonify(final_output)

if __name__=="__main__":
    app.run(debug=True)