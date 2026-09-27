import gradio as gr
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')

def predict_churn(tenure, monthly_charges, total_charges, contract):
    data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract]
    })
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    return "Churn" if prediction == 1 else "No Churn", f"{prob:.2f}"

inputs = [
    gr.Number(label="Tenure (months)"),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges"),
    gr.Dropdown(['Month-to-month', 'One year', 'Two year'], label="Contract")
]

outputs = [gr.Textbox(label="Prediction"), gr.Textbox(label="Churn Probability")]

gr.Interface(fn=predict_churn, inputs=inputs, outputs=outputs, title="Customer Churn Prediction").launch()
