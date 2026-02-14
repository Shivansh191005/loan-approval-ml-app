#backend website

# streamlit = library to create UI
import streamlit as st

# numpy = handle numeric arrays
import numpy as np

# pickle = load trained models
import pickle

import pandas as pd

# ---------------- LOAD MODELS ----------------

# dictionary storing all models
models = {
    "Logistic Regression": pickle.load(open("logistic.pkl","rb")),
    "Decision Tree": pickle.load(open("decision_tree.pkl","rb")),
    "Random Forest": pickle.load(open("random_forest.pkl","rb"))
}

# load scaler used during training
scaler = pickle.load(open("scaler.pkl","rb"))


# ---------------- PAGE TITLE ----------------
st.title("🏦 Loan Approval Prediction System")


# ---------------- MODEL SELECTION ----------------
# dropdown menu to choose model
model_choice = st.selectbox("Choose Model", list(models.keys()))

# selected model
model = models[model_choice]


# ---------------- USER INPUT FIELDS ----------------
# dropdowns and boxes for user input

gender = st.selectbox("Gender", ["Male","Female"])
married = st.selectbox("Married", ["Yes","No"])
dependents = st.selectbox("Dependents", [0,1,2,3])
education = st.selectbox("Education", ["Graduate","Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes","No"])

app_income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")

credit_history = st.selectbox("Credit History", [1,0])
property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])


# ---------------- DATA PREPROCESSING ----------------
# convert text into numbers (same as training)

gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
education = 1 if education=="Graduate" else 0
self_employed = 1 if self_employed=="Yes" else 0

if property_area=="Urban":
    property_area=2
elif property_area=="Semiurban":
    property_area=1
else:
    property_area=0


# create dataframe with same column names as training
features = pd.DataFrame([[
    gender, married, dependents, education, self_employed,
    app_income, co_income, loan_amount, loan_term,
    credit_history, property_area
]], columns=[
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
    "Credit_History","Property_Area"
])

# scale properly
features = scaler.transform(features)

# ---------------- PREDICTION BUTTON ----------------
prediction = model.predict(features)
prob = model.predict_proba(features)[0][1]   # probability of approval

if prediction[0]==1:
    st.success("✅ Loan Approved")
    st.write(f"Approval Confidence: {prob*100:.2f}%")
else:
    st.error("❌ Loan Rejected")
    st.write(f"Approval Confidence: {prob*100:.2f}%")

st.subheader("Decision Factors")

if credit_history == 0:
    st.write("• Poor credit history reduces approval chances")
if app_income < loan_amount*10:
    st.write("• Low income compared to loan amount")
if self_employed == 1:
    st.write("• Self employment may increase risk")
