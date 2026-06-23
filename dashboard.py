import streamlit as st
import pandas as pd
import plotly.express as px

# Load Dataset
df = pd.read_csv("cleaned_titanic.csv")

st.title("Titanic Data Science Dashboard")

# Dataset Overview
st.subheader("Dataset Overview")
st.write(df.head())

# Survival Count
fig1 = px.histogram(
    df,
    x="Survived",
    title="Survival Count"
)
st.plotly_chart(fig1)

# Gender vs Survival
fig2 = px.histogram(
    df,
    x="Sex",
    color="Survived",
    barmode="group",
    title="Gender vs Survival"
)
st.plotly_chart(fig2)

# Passenger Class vs Survival
fig3 = px.histogram(
    df,
    x="Pclass",
    color="Survived",
    barmode="group",
    title="Passenger Class vs Survival"
)
st.plotly_chart(fig3)

# Age Distribution
fig4 = px.histogram(
    df,
    x="Age",
    nbins=30,
    title="Age Distribution"
)
st.plotly_chart(fig4)

# Fare Distribution
fig5 = px.histogram(
    df,
    x="Fare",
    nbins=30,
    title="Fare Distribution"
)
st.plotly_chart(fig5)

st.success("Titanic Dashboard Loaded Successfully!")
