import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import io

def show_historical_data(patient_id):
    """
    Display and analyze historical patient data
    
    Args:
        patient_id (str): ID of the selected patient
    """
    st.title("Historical Health Data")
    
    # Get patient info and data
    patient_info = st.session_state.patients[patient_id]
    patient_data = st.session_state.patient_data[patient_id].copy()
    
    # Add a date column for easier filtering
    patient_data['date'] = patient_data['timestamp'].dt.date
    
    st.subheader(f"{patient_info['name']}'s Health Records")
    
    # Date range filter
    st.markdown("### Select Date Range")
    col1, col2 = st.columns(2)
    
    min_date = patient_data['date'].min()
    max_date = patient_data['date'].max()
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Filter data based on date range
    filtered_data = patient_data[(patient_data['date'] >= start_date) & 
                                 (patient_data['date'] <= end_date)]
    
    # Parameter selection for detailed view
    st.markdown("### Select Vital Signs to View")
    
    parameters = st.multiselect(
        "Parameters",
        options=[
            "heart_rate", 
            "blood_pressure_systolic", 
            "blood_pressure_diastolic", 
            "temperature", 
            "oxygen_saturation", 
            "respiratory_rate",
            "glucose"
        ],
        default=["heart_rate", "blood_pressure_systolic", "oxygen_saturation"]
    )
    
    # Display interactive chart based on selection
    if parameters:
        st.markdown("### Detailed Vital Signs Chart")
        
        # Create figure with multiple y-axes if needed
        if len(parameters) > 1:
            # For multiple parameters, use a more sophisticated approach
            fig = go.Figure()
            
            colors = px.colors.qualitative.Plotly
            
            for i, param in enumerate(parameters):
                fig.add_trace(go.Scatter(
                    x=filtered_data['timestamp'],
                    y=filtered_data[param],
                    name=param.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title="Historical Vital Signs",
                xaxis_title="Time",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )
            
        else:
            # For a single parameter, simple line chart
            param = parameters[0]
            
            fig = px.line(
                filtered_data,
                x='timestamp',
                y=param,
                title=f"Historical {param.replace('_', ' ').title()}",
                labels={
                    'timestamp': 'Time',
                    param: param.replace('_', ' ').title()
                },
                height=500
            )
            
            # Add reference lines based on parameter
            if param == 'heart_rate':
                fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
                fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Upper Normal")
            elif param == 'blood_pressure_systolic':
                fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
                fig.add_hline(y=140, line_dash="dash", line_color="orange", annotation_text="Upper Normal")
            elif param == 'blood_pressure_diastolic':
                fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
                fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Upper Normal")
            elif param == 'temperature':
                fig.add_hline(y=36.0, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
                fig.add_hline(y=38.0, line_dash="dash", line_color="orange", annotation_text="Upper Normal")
            elif param == 'oxygen_saturation':
                fig.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
            elif param == 'respiratory_rate':
                fig.add_hline(y=12, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
                fig.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Upper Normal")
            elif param == 'glucose':
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Lower Normal")
                fig.add_hline(y=140, line_dash="dash", line_color="orange", annotation_text="Upper Normal")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary of the data
    st.markdown("### Statistical Summary")
    
    # Get daily stats
    daily_stats = filtered_data.groupby('date').agg({
        'heart_rate': ['mean', 'min', 'max', 'std'],
        'blood_pressure_systolic': ['mean', 'min', 'max', 'std'],
        'blood_pressure_diastolic': ['mean', 'min', 'max', 'std'],
        'temperature': ['mean', 'min', 'max', 'std'],
        'oxygen_saturation': ['mean', 'min', 'max', 'std'],
        'respiratory_rate': ['mean', 'min', 'max', 'std'],
        'glucose': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Flatten the multi-index columns
    daily_stats.columns = ['_'.join(col).strip('_') for col in daily_stats.columns.values]
    
    # Display the stats in an expandable section
    with st.expander("View Daily Statistics", expanded=False):
        st.dataframe(daily_stats)
    
    # Data export functionality
    st.markdown("### Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "Excel"]
        )
    
    with col2:
        if st.button("Export Data"):
            if export_format == "CSV":
                csv = filtered_data.to_csv(index=False)
                
                # Create a download link for the CSV
                csv_bytes = csv.encode()
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=f"{patient_info['name'].replace(' ', '_')}_health_data.csv",
                    mime="text/csv"
                )
                
            else:  # Excel
                # Use BytesIO to create virtual Excel file
                buffer = io.BytesIO()
                
                # Create Excel writer
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Write the filtered data to the Excel file
                    filtered_data.to_excel(writer, sheet_name="Health Data", index=False)
                    
                    # Write daily stats to a separate sheet
                    daily_stats.to_excel(writer, sheet_name="Daily Statistics", index=False)
                    
                    # Get the workbook
                    workbook = writer.book
                    
                    # Add a chart sheet for visualizing data
                    chart_sheet = workbook.add_worksheet("Data Charts")
                    
                    # Create a chart for heart rate
                    chart = workbook.add_chart({'type': 'line'})
                    
                    # Configure the chart
                    chart.add_series({
                        'name': 'Heart Rate',
                        'categories': '=Health Data!$A$2:$A$' + str(len(filtered_data) + 1),
                        'values': '=Health Data!$C$2:$C$' + str(len(filtered_data) + 1),
                    })
                    
                    # Add the chart to the chart sheet
                    chart_sheet.insert_chart('A1', chart, {'x_scale': 2, 'y_scale': 2})
                
                # Get the Excel file as bytes
                excel_bytes = buffer.getvalue()
                
                # Create a download button for the Excel file
                st.download_button(
                    label="Download Excel",
                    data=excel_bytes,
                    file_name=f"{patient_info['name'].replace(' ', '_')}_health_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # View raw data with filtering options
    st.markdown("### Raw Data View")
    
    # Let the user decide how many rows to show
    num_rows = st.slider("Number of rows to display", 10, 100, 20)
    
    # Add a search filter
    search_term = st.text_input("Search in data")
    
    # Filter based on search term if provided
    if search_term:
        # Search across all columns
        search_mask = np.column_stack([
            filtered_data[col].astype(str).str.contains(search_term, case=False, na=False)
            for col in filtered_data.columns
        ]).any(axis=1)
        
        search_results = filtered_data[search_mask]
        st.dataframe(search_results.head(num_rows))
        st.write(f"Found {len(search_results)} matching records.")
    else:
        st.dataframe(filtered_data.head(num_rows))
        st.write(f"Showing {min(num_rows, len(filtered_data))} of {len(filtered_data)} records.")
