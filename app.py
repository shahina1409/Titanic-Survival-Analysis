import streamlit as st
import pickle
import pandas as pd

# Load Model
model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 100, 25)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
family_size = st.number_input("Family Size", 1, 10, 1)
is_alone = st.selectbox("Is Alone", [0, 1])
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

# Encoding
sex = 0 if sex == "Male" else 1

embarked_map = {"S": 0, "C": 1, "Q": 2}
embarked = embarked_map[embarked]

title_map = {
    "Mr": 0,
    "Miss": 1,
    "Mrs": 2,
    "Master": 3,
    "Rare": 4
}
title = title_map[title]

# Prediction
if st.button("Predict"):

    data = pd.DataFrame([[
        pclass,
        sex,
        age,
        fare,
        family_size,
        is_alone,
        embarked,
        title
    ]], columns=[
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "FamilySize",
        "IsAlone",
        "Embarked",
        "Title"
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Passenger is likely to Survive ✅")
    else:
        st.error("Passenger is unlikely to Survive ❌")