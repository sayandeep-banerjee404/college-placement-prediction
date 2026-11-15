import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('placement_prediction_model.pkl')

# Page configuration
st.set_page_config(
    page_title="College Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 College Placement Prediction System")
st.write("Predict the placement rate of a college using Machine Learning.")

st.divider()

st.subheader("Enter College Details")

# Input fields
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)
    college_tier = st.selectbox(
        "College Tier", ["Tier 1", "Tier 2", "Tier 3"]
    )
    total_students = st.number_input(
        "Total Students", min_value=1, value=500
    )
    eligible_students = st.number_input(
        "Eligible Students", min_value=1, value=400
    )

with col2:
    companies_visited = st.number_input(
        "Companies Visited", min_value=0, value=30
    )
    average_cgpa = st.number_input(
        "Average CGPA", min_value=0.0, max_value=10.0,
        value=7.5, step=0.1
    )
    internship_rate = st.number_input(
        "Internship Rate (%)", min_value=0.0,
        max_value=100.0, value=60.0
    )
    industry_demand = st.number_input(
        "Industry Demand Index", min_value=0.0,
        max_value=100.0, value=70.0
    )

st.divider()

# Prediction button
if st.button("Predict Placement Rate", type="primary"):

    if eligible_students > total_students:
        st.error("Eligible students cannot exceed total students.")

    else:
        input_data = pd.DataFrame([{
            'Year': year,
            'College_Tier': college_tier,
            'Total_Students': total_students,
            'Eligible_Students': eligible_students,
            'Companies_Visited': companies_visited,
            'Average_CGPA': average_cgpa,
            'Internship_Rate_Percent': internship_rate,
            'Industry_Demand_Index': industry_demand
        }])

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(100, prediction))

        st.success("Prediction Completed!")

        st.metric(
            "Predicted Placement Rate",
            f"{prediction:.2f}%"
        )

        st.progress(int(prediction))

        st.write(
            "This prediction is generated using the trained "
            "Random Forest Machine Learning model."
        )