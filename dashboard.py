import streamlit as st
import pandas as pd
import model_training

st.title("Safety Observations Dashboard")

# Load data and predictions
data = pd.read_csv('sample_data.csv')
model, _, _ = model_training.train_model()
predictions = model.predict(data['PNT_ATRISKNOTES_TX'])

# Display results
high_risk_comments = data[predictions == "Critical"]
st.write(f"High-Risk Comments: {len(high_risk_comments)}")
st.table(high_risk_comments[['PNT_ATRISKNOTES_TX']])
