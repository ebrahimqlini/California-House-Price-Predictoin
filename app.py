import pickle
from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # استلام البيانات
    data = request.json['data']
    
    # تحويل البيانات لـ DataFrame عشان نحافظ على أسماء الأعمدة ونتجنب الـ Warning
    # الترتيب هنا بيعتمد على ترتيب الـ Keys اللي بتبعتها في الـ JSON
    df_input = pd.DataFrame([list(data.values())], columns=list(data.keys()))
    
    # عمل Scaling
    new_data = scalar.transform(df_input)
    
    # التوقع
    output = regmodel.predict(new_data)
    
    # تقريب النتيجة لـ 4 أرقام عشرية
    final_output = round(float(output[0]), 4)
    
    print(f"Prediction: {final_output}") # عشان تشوفها في التيرمينال برضه
    
    return jsonify(final_output)

if __name__=="__main__":
    app.run(debug=True)
