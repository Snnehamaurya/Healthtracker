import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from models.anomaly_detection import detect_anomalies
from utils.data_processing import generate_example_patient_data, process_vital_data
from utils.auth import initialize_auth_state, show_login_page, logout_user

# Set page configuration
st.set_page_config(
    page_title="Health Monitor AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication state
initialize_auth_state()

# Initialize session state variables if they don't exist
if 'patients' not in st.session_state:
    # Initialize with sample patient data
    st.session_state.patients = {
        "P001": {"name": "John Doe", "age": 45, "gender": "Male", "condition": "Hypertension"},
        "P002": {"name": "Jane Smith", "age": 52, "gender": "Female", "condition": "Diabetes"},
        "P003": {"name": "Michael Johnson", "age": 63, "gender": "Male", "condition": "Heart Disease"},
        "P004": {"name": "Emma Wilson", "age": 35, "gender": "Female", "condition": "Asthma"}
    }

if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = "P001"

if 'patient_data' not in st.session_state:
    # Generate example data for each patient
    st.session_state.patient_data = {}
    for patient_id in st.session_state.patients:
        st.session_state.patient_data[patient_id] = generate_example_patient_data(
            days=30, 
            condition=st.session_state.patients[patient_id]["condition"]
        )

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'show_registration' not in st.session_state:
    st.session_state.show_registration = False

# Check if user is logged in
if not st.session_state.logged_in:
    # Display login page
    if show_login_page():
        st.rerun()
else:
    # Display main app after successful login
    # Sidebar for navigation
    st.sidebar.title("Health Monitor AI")
    
    # Display logged in user info
    st.sidebar.markdown(f"Logged in as: **{st.session_state.user_name}** ({st.session_state.user_role})")
    
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()
        
    st.sidebar.markdown("---")

    # Patient selection
    st.sidebar.subheader("Patient Selection")
    selected_patient_id = st.sidebar.selectbox(
        "Select Patient",
        options=list(st.session_state.patients.keys()),
        format_func=lambda x: f"{x}: {st.session_state.patients[x]['name']}",
        key="patient_selector"
    )

    st.session_state.selected_patient = selected_patient_id
    patient_info = st.session_state.patients[selected_patient_id]

    # Display patient information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Patient Information")
    st.sidebar.write(f"**Name:** {patient_info['name']}")
    st.sidebar.write(f"**Age:** {patient_info['age']}")
    st.sidebar.write(f"**Gender:** {patient_info['gender']}")
    st.sidebar.write(f"**Condition:** {patient_info['condition']}")

    # Navigation menu
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    
    # Store navigation selection in session state
    if 'nav_selection' not in st.session_state:
        st.session_state.nav_selection = "Dashboard"
    
    # Use the stored selection as the default value
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Patient Profile", "Historical Data", "Analytics", "Health Assistant"],
        key="page_nav"
    )
    
    # Update the stored selection
    st.session_state.nav_selection = page

    # Main content area
    if page == "Dashboard":
        # Import and display dashboard
        from pages.dashboard import show_dashboard
        show_dashboard(selected_patient_id)
        
    elif page == "Patient Profile":
        # Import and display patient profile
        from pages.patient_profile import show_patient_profile
        show_patient_profile(selected_patient_id)
        
    elif page == "Historical Data":
        # Import and display historical data
        from pages.historical_data import show_historical_data
        show_historical_data(selected_patient_id)
        
    elif page == "Analytics":
        # Import and display analytics
        from pages.analytics import show_analytics
        show_analytics(selected_patient_id)
        
    elif page == "Health Assistant":
        # Import and display chatbot
        from pages.chatbot import show_chatbot
        show_chatbot(selected_patient_id)

    # Alert system in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alerts")

    # Run anomaly detection on the current patient's data
    patient_data = st.session_state.patient_data[selected_patient_id]
    latest_data = patient_data.iloc[-1:]  # Get the most recent data point

    # Detect anomalies in the latest data
    anomalies = detect_anomalies(latest_data, patient_data.iloc[:-1])

    # Display alerts if any anomalies are detected
    if anomalies:
        for feature, is_anomaly in anomalies.items():
            if is_anomaly:
                alert_message = f"Anomaly detected in {feature} for {patient_info['name']}"
                if alert_message not in st.session_state.alerts:
                    st.session_state.alerts.append(alert_message)

    # Display all alerts
    if st.session_state.alerts:
        for alert in st.session_state.alerts:
            st.sidebar.error(alert)
        
        if st.sidebar.button("Clear Alerts"):
            st.session_state.alerts = []
            st.rerun()
    else:
        st.sidebar.success("No active alerts")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Health Monitor AI v1.0.0")
