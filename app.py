import json
import pickle
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return model, metrics


model, metrics = load_model()

st.title("Loan Approval Predictor")

st.sidebar.header("Applicant inputs")

def user_inputs():
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
    Married = st.sidebar.selectbox("Married", ["Yes", "No"], index=0)
    Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
    Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
    Self_Employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"], index=0)
    Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)
    Home_Ownership = st.sidebar.selectbox("Home Ownership", ["Own", "Mortgage", "Rent"], index=0)
    Loan_Purpose = st.sidebar.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Car", "Business", "Debt Consolidation"], index=0)

    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
    Applicant_Income = st.sidebar.number_input("Applicant Income($)", min_value=0, value=5000)
    Coapplicant_Income = st.sidebar.number_input("Coapplicant Income($)", min_value=0, value=0)
    Loan_Amount = st.sidebar.number_input("Loan Amount($000)", min_value=0, value=100)
    Loan_Term = st.sidebar.number_input("Loan Term(months)", min_value=12, max_value=480, value=360)
    Credit_History = st.sidebar.selectbox("Credit History", [0, 1], index=1)
    Credit_Score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=650)
    Employment_Years = st.sidebar.number_input("Employment Years", min_value=0, max_value=60, value=5)
    Existing_Loans = st.sidebar.number_input("Existing Loans", min_value=0, max_value=10, value=0)

    data = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self Employed": Self_Employed,
        "Property Area": Property_Area,
        "Home Ownership": Home_Ownership,
        "Loan Purpose": Loan_Purpose,
        "Age": Age,
        "Applicant Income($)": Applicant_Income,
        "Coapplicant Income($)": Coapplicant_Income,
        "Loan Amount($000)": Loan_Amount,
        "Loan Term(months)": Loan_Term,
        "Credit History": Credit_History,
        "Credit Score": Credit_Score,
        "Employment Years": Employment_Years,
        "Existing Loans": Existing_Loans,
    }

    return pd.DataFrame([data])


input_df = user_inputs()

st.subheader("Model metrics")
st.write(metrics)

if st.button("Predict Loan Approval"):
    # fix Dependents format for consistency with training
    input_df["Dependents"] = input_df["Dependents"].replace({"3+": "3"}).astype(float)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"Approval probability: {prob:.3f}")
    st.write("Approved" if int(pred) == 1 else "Not Approved")
