import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.title("Customer Churn Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

@st.cache_resource
def train_model():
    df = load_data()

    # Drop columns not used
    df = df.drop(["id", "CustomerId", "Surname"], axis=1)

    # Convert categorical
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, X.columns

model, scaler, feature_columns = train_model()

st.sidebar.header("Customer Details")

def user_input():
    data = {
        "CreditScore": st.sidebar.slider("Credit Score", 300, 900, 600),
        "Age": st.sidebar.slider("Age", 18, 92, 40),
        "Tenure": st.sidebar.slider("Tenure", 0, 10, 3),
        "Balance": st.sidebar.number_input("Balance", 0.0, 250000.0, 60000.0),
        "NumOfProducts": st.sidebar.slider("Products", 1, 4, 2),
        "HasCrCard": st.sidebar.selectbox("Has Card", [0, 1]),
        "IsActiveMember": st.sidebar.selectbox("Active Member", [0, 1]),
        "EstimatedSalary": st.sidebar.number_input("Salary", 1000.0, 200000.0, 50000.0)
    }
    return pd.DataFrame([data])

input_df = user_input()

# Align columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    if pred == 1:
        st.error("Customer will CHURN")
    else:
        st.success("Customer will STAY")
