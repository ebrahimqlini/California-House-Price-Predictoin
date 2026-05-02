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
    
    # Print the result to the terminal for debugging purposes
    print(f"Prediction: {output[0]}")
    
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)