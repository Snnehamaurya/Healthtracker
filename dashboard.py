import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import process_vital_data

def show_dashboard(patient_id):
    """
    Display the main dashboard for patient monitoring
    
    Args:
        patient_id (str): ID of the selected patient
    """
    st.title("Patient Health Dashboard")
    
    # Get patient info and data
    patient_info = st.session_state.patients[patient_id]
    patient_data = st.session_state.patient_data[patient_id]
    
    # Process the latest data
    latest_data = patient_data.iloc[-24:]  # Last 24 hours
    vital_analysis = process_vital_data(latest_data)
    
    # Display patient name and current status
    st.header(f"{patient_info['name']} - {patient_info['condition']}")
    
    # Current vital signs - top row with gauge charts
    st.subheader("Current Vital Signs")
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest readings
    latest = patient_data.iloc[-1]
    
    with col1:
        create_gauge(
            title="Heart Rate", 
            value=int(latest["heart_rate"]), 
            min_val=40, max_val=160, 
            thresholds={"low": 60, "high": 100},
            unit="bpm"
        )
        
    with col2:
        create_gauge(
            title="Blood Pressure", 
            value=int(latest["blood_pressure_systolic"]), 
            min_val=80, max_val=200, 
            thresholds={"low": 90, "high": 140},
            unit="mmHg",
            subtext=f"{int(latest['blood_pressure_diastolic'])} mmHg diastolic"
        )
        
    with col3:
        create_gauge(
            title="Oxygen Saturation", 
            value=int(latest["oxygen_saturation"]), 
            min_val=70, max_val=100, 
            thresholds={"low": 95, "high": 101},
            unit="%",
            decreasing=True
        )
        
    with col4:
        create_gauge(
            title="Temperature", 
            value=round(latest["temperature"], 1), 
            min_val=35, max_val=41, 
            thresholds={"low": 36.0, "high": 38.0},
            unit="°C"
        )
    
    # Second row of gauges
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_gauge(
            title="Respiratory Rate", 
            value=round(latest["respiratory_rate"], 1), 
            min_val=8, max_val=30, 
            thresholds={"low": 12, "high": 20},
            unit="BrPM"
        )
        
    with col2:
        create_gauge(
            title="Glucose Level", 
            value=int(latest["glucose"]), 
            min_val=40, max_val=250, 
            thresholds={"low": 70, "high": 140},
            unit="mg/dL"
        )
    
    # Display summary of vital signs status
    st.markdown("---")
    st.subheader("Current Health Assessment")
    
    assessment_cols = st.columns(4)
    
    # Heart Rate Assessment
    with assessment_cols[0]:
        status = vital_analysis["summary"]["heart_rate"]
        icon = "✅" if status == "Normal" else "⚠️"
        st.markdown(f"**Heart Rate**: {icon} {status}")
    
    # Blood Pressure Assessment
    with assessment_cols[1]:
        status = vital_analysis["summary"]["blood_pressure"]
        icon = "✅" if status == "Normal" else "⚠️"
        st.markdown(f"**Blood Pressure**: {icon} {status}")
    
    # Oxygen Assessment
    with assessment_cols[2]:
        status = vital_analysis["summary"]["oxygen_saturation"]
        icon = "✅" if status == "Normal" else "⚠️"
        st.markdown(f"**Oxygen Saturation**: {icon} {status}")
    
    # Temperature Assessment
    with assessment_cols[3]:
        status = vital_analysis["summary"]["temperature"]
        icon = "✅" if status == "Normal" else "⚠️"
        st.markdown(f"**Temperature**: {icon} {status}")
    
    # Display trend charts for the last 24 hours
    st.markdown("---")
    st.subheader("Vital Signs Trends (Last 24 Hours)")
    
    # Time window selection
    time_window = st.selectbox(
        "Time Window",
        options=["Last 24 Hours", "Last 48 Hours", "Last Week"],
        index=0
    )
    
    if time_window == "Last 24 Hours":
        display_data = patient_data.iloc[-24:]
    elif time_window == "Last 48 Hours":
        display_data = patient_data.iloc[-48:]
    else:  # Last Week
        display_data = patient_data.iloc[-168:]  # 7 days * 24 hours
    
    # Create tabs for different vital signs
    tabs = st.tabs(["Heart Rate", "Blood Pressure", "Oxygen & Respiratory", "Temperature & Glucose"])
    
    # Heart Rate Tab
    with tabs[0]:
        fig = px.line(
            display_data, 
            x="timestamp", 
            y="heart_rate",
            labels={"timestamp": "Time", "heart_rate": "Heart Rate (bpm)"},
            title="Heart Rate Trend"
        )
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Lower Limit")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Upper Limit")
        st.plotly_chart(fig, use_container_width=True)
    
    # Blood Pressure Tab
    with tabs[1]:
        fig = px.line(
            display_data, 
            x="timestamp", 
            y=["blood_pressure_systolic", "blood_pressure_diastolic"],
            labels={"timestamp": "Time", "value": "Blood Pressure (mmHg)", "variable": "Type"},
            title="Blood Pressure Trend",
            color_discrete_map={
                "blood_pressure_systolic": "red",
                "blood_pressure_diastolic": "blue"
            }
        )
        fig.add_hline(y=140, line_dash="dash", line_color="red", annotation_text="Systolic Upper Limit")
        fig.add_hline(y=90, line_dash="dash", line_color="blue", annotation_text="Diastolic Upper Limit")
        st.plotly_chart(fig, use_container_width=True)
    
    # Oxygen & Respiratory Tab
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                display_data, 
                x="timestamp", 
                y="oxygen_saturation",
                labels={"timestamp": "Time", "oxygen_saturation": "SpO2 (%)"},
                title="Oxygen Saturation Trend"
            )
            fig.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="Lower Limit")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.line(
                display_data, 
                x="timestamp", 
                y="respiratory_rate",
                labels={"timestamp": "Time", "respiratory_rate": "Breaths per Minute"},
                title="Respiratory Rate Trend"
            )
            fig.add_hline(y=12, line_dash="dash", line_color="orange", annotation_text="Lower Limit")
            fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Upper Limit")
            st.plotly_chart(fig, use_container_width=True)
    
    # Temperature & Glucose Tab
    with tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                display_data, 
                x="timestamp", 
                y="temperature",
                labels={"timestamp": "Time", "temperature": "Temperature (°C)"},
                title="Body Temperature Trend"
            )
            fig.add_hline(y=36.0, line_dash="dash", line_color="blue", annotation_text="Lower Limit")
            fig.add_hline(y=38.0, line_dash="dash", line_color="red", annotation_text="Upper Limit")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.line(
                display_data, 
                x="timestamp", 
                y="glucose",
                labels={"timestamp": "Time", "glucose": "Glucose (mg/dL)"},
                title="Blood Glucose Trend"
            )
            fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Lower Limit")
            fig.add_hline(y=140, line_dash="dash", line_color="orange", annotation_text="Upper Limit")
            st.plotly_chart(fig, use_container_width=True)

def create_gauge(title, value, min_val, max_val, thresholds, unit="", subtext="", decreasing=False):
    """
    Create and display a gauge chart for vital signs
    
    Args:
        title (str): Title of the gauge
        value (float): Current value to display
        min_val (float): Minimum value on the gauge
        max_val (float): Maximum value on the gauge
        thresholds (dict): Dict with 'low' and 'high' threshold values
        unit (str): Unit of measurement
        subtext (str): Additional text to display below the gauge
        decreasing (bool): If True, values below threshold are concerning (reverse color logic)
    """
    # Define colors based on thresholds
    if decreasing:
        # For metrics where lower is worse (like oxygen)
        if value < thresholds["low"]:
            color = "red"
        else:
            color = "green"
    else:
        # For standard metrics where extremes are concerning
        if value < thresholds["low"] or value > thresholds["high"]:
            color = "red"
        else:
            color = "green"
    
    # Create figure with gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": color},
            "steps": [
                {"range": [min_val, thresholds["low"]], "color": "lightgray"},
                {"range": [thresholds["low"], thresholds["high"]], "color": "lightgreen"},
                {"range": [thresholds["high"], max_val], "color": "lightgray"}
            ],
        },
        number={"suffix": f" {unit}", "font": {"size": 26, "color": color}}
    ))
    
    # Update layout
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Display the gauge
    st.plotly_chart(fig, use_container_width=True)
    
    if subtext:
        st.caption(subtext)
