import streamlit as st
import joblib
import pandas as pd

# Page settings
st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨")

# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("🏨 Hotel Booking Risk Checker")

st.write("Enter simple booking details to estimate if a customer might cancel.")

st.markdown("---")

# Layout (two columns for cleaner UI)
col1, col2 = st.columns(2)

with col1:
    lead_time = st.slider("Days before check-in (how early the booking was made)", 0, 500, 30)
    adults = st.number_input("Number of adults", 1, 10, 2)
    special_requests = st.slider("Special requests (e.g., extra bed, late check-in)", 0, 5, 0)

with col2:
    price = st.number_input("Room price per night", 0, 1000, 100)
    children = st.number_input("Number of children", 0, 10, 0)

st.markdown("---")

# Predict button
if st.button("🔍 Check Booking Risk"):

    input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

    # Fill values (mapping to model features)
    if 'lead_time' in input_data:
        input_data['lead_time'] = lead_time

    if 'avg_price_per_room' in input_data:
        input_data['avg_price_per_room'] = price

    if 'no_of_adults' in input_data:
        input_data['no_of_adults'] = adults

    if 'no_of_children' in input_data:
        input_data['no_of_children'] = children

    if 'no_of_special_requests' in input_data:
        input_data['no_of_special_requests'] = special_requests

    # Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.markdown("## Result")

    if prediction == "Canceled":
        st.error("⚠️ High risk: This booking may be canceled")
        st.progress(int(prob[0]*100))
        st.write(f"Estimated cancellation risk: **{round(prob[0]*100, 2)}%**")
    else:
        st.success("✅ Low risk: This booking is likely to be confirmed")
        st.progress(int(prob[1]*100))
        st.write(f"Estimated confirmation probability: **{round(prob[1]*100, 2)}%**")

    st.info("Tip: Early bookings with fewer special requests are more likely to be canceled.")