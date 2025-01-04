import pickle
from flask import Flask,request,jsonify
import numpy as np
import pandas as pd



app = Flask(__name__)



with open('diabates_model.pkl','rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl','rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route("/")

def home():
    return "Diabetes Prediction App is running"


@app.route("/predict",methods=['POST'])
def predict():
    try:
        #get the JSON data to our API request
        data = request.get_json()

        input_data = pd.DataFrame([data])

        #checkl if input provided
        if not data:
            return jsonify({"error": "No input provided"}), 400
        #validate input fields
        required_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                         'Insulin','BMI','DiabetesPedigreeFunction','Age']
        if not all(col in input_data.columns for col in required_cols):
            return jsonify({"error": f"Missing required fields. Required columns: {required_cols}"}), 400
        

        #scale the input
        scaled_input = scaler.transform(input_data)

        #make prediction
        prediction = model.predict(scaled_input)

        #response
        response = {
            "prediction": "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
if __name__ == "__main__":
    app.run(debug=True)
