import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_example_patient_data(days=30, condition="Normal"):
    """
    Generate example patient vital signs data for demonstration purposes.
    
    Args:
        days (int): Number of days of data to generate
        condition (str): Medical condition to simulate
        
    Returns:
        pd.DataFrame: DataFrame with simulated vital signs data
    """
    # Create timestamp range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Base parameters for normal ranges
    base_heart_rate = 75
    base_bp_systolic = 120
    base_bp_diastolic = 80
    base_temperature = 36.8
    base_oxygen = 98
    base_respiratory = 14
    base_glucose = 100
    
    # Create random variation based on condition
    np.random.seed(42)  # For reproducibility
    
    if condition == "Hypertension":
        # Higher blood pressure for hypertension
        bp_systolic_values = np.random.normal(base_bp_systolic + 30, 10, len(timestamps))
        bp_diastolic_values = np.random.normal(base_bp_diastolic + 15, 8, len(timestamps))
        heart_rate_values = np.random.normal(base_heart_rate + 5, 8, len(timestamps))
        temperature_values = np.random.normal(base_temperature, 0.3, len(timestamps))
        oxygen_values = np.random.normal(base_oxygen - 1, 1, len(timestamps))
        respiratory_values = np.random.normal(base_respiratory + 1, 2, len(timestamps))
        glucose_values = np.random.normal(base_glucose, 15, len(timestamps))
        
    elif condition == "Diabetes":
        # Fluctuating glucose for diabetes
        bp_systolic_values = np.random.normal(base_bp_systolic + 10, 12, len(timestamps))
        bp_diastolic_values = np.random.normal(base_bp_diastolic + 5, 10, len(timestamps))
        heart_rate_values = np.random.normal(base_heart_rate, 10, len(timestamps))
        temperature_values = np.random.normal(base_temperature, 0.3, len(timestamps))
        oxygen_values = np.random.normal(base_oxygen, 1.5, len(timestamps))
        respiratory_values = np.random.normal(base_respiratory, 2, len(timestamps))
        glucose_values = np.random.normal(base_glucose + 60, 30, len(timestamps))
        
    elif condition == "Heart Disease":
        # Irregular heart rate for heart disease
        heart_rate_pattern = np.sin(np.linspace(0, 10*np.pi, len(timestamps))) * 15
        bp_systolic_values = np.random.normal(base_bp_systolic + 15, 15, len(timestamps))
        bp_diastolic_values = np.random.normal(base_bp_diastolic + 10, 10, len(timestamps))
        heart_rate_values = np.random.normal(base_heart_rate + heart_rate_pattern, 12, len(timestamps))
        temperature_values = np.random.normal(base_temperature, 0.2, len(timestamps))
        oxygen_values = np.random.normal(base_oxygen - 3, 2, len(timestamps))
        respiratory_values = np.random.normal(base_respiratory + 2, 3, len(timestamps))
        glucose_values = np.random.normal(base_glucose + 10, 15, len(timestamps))
        
    elif condition == "Asthma":
        # Lower oxygen and higher respiratory rate for asthma
        respiratory_pattern = np.sin(np.linspace(0, 5*np.pi, len(timestamps))) * 3
        bp_systolic_values = np.random.normal(base_bp_systolic, 10, len(timestamps))
        bp_diastolic_values = np.random.normal(base_bp_diastolic, 8, len(timestamps))
        heart_rate_values = np.random.normal(base_heart_rate + 5, 10, len(timestamps))
        temperature_values = np.random.normal(base_temperature, 0.3, len(timestamps))
        oxygen_values = np.random.normal(base_oxygen - 5, 3, len(timestamps))
        respiratory_values = np.random.normal(base_respiratory + 6 + respiratory_pattern, 3, len(timestamps))
        glucose_values = np.random.normal(base_glucose, 10, len(timestamps))
        
    else:  # Normal condition
        bp_systolic_values = np.random.normal(base_bp_systolic, 8, len(timestamps))
        bp_diastolic_values = np.random.normal(base_bp_diastolic, 6, len(timestamps))
        heart_rate_values = np.random.normal(base_heart_rate, 7, len(timestamps))
        temperature_values = np.random.normal(base_temperature, 0.2, len(timestamps))
        oxygen_values = np.random.normal(base_oxygen, 1, len(timestamps))
        respiratory_values = np.random.normal(base_respiratory, 1.5, len(timestamps))
        glucose_values = np.random.normal(base_glucose, 10, len(timestamps))
    
    # Add a simulated anomaly for the last data point (with 20% probability)
    if np.random.random() < 0.2:
        feature_to_modify = np.random.choice([
            'heart_rate_values', 'bp_systolic_values', 'bp_diastolic_values', 
            'temperature_values', 'oxygen_values', 'respiratory_values'
        ])
        
        # Get the locals dictionary to access the variable by name
        locals_dict = locals()
        
        # Apply a significant deviation to the chosen feature
        if feature_to_modify == 'heart_rate_values':
            locals_dict[feature_to_modify][-1] = locals_dict[feature_to_modify][-1] * 1.5
        elif feature_to_modify == 'bp_systolic_values':
            locals_dict[feature_to_modify][-1] = locals_dict[feature_to_modify][-1] * 1.4
        elif feature_to_modify == 'bp_diastolic_values':
            locals_dict[feature_to_modify][-1] = locals_dict[feature_to_modify][-1] * 1.4
        elif feature_to_modify == 'temperature_values':
            locals_dict[feature_to_modify][-1] = locals_dict[feature_to_modify][-1] * 1.1
        elif feature_to_modify == 'oxygen_values':
            locals_dict[feature_to_modify][-1] = locals_dict[feature_to_modify][-1] * 0.8
        elif feature_to_modify == 'respiratory_values':
            locals_dict[feature_to_modify][-1] = locals_dict[feature_to_modify][-1] * 1.6
    
    # Apply physical constraints
    heart_rate_values = np.clip(heart_rate_values, 40, 200)
    bp_systolic_values = np.clip(bp_systolic_values, 70, 220)
    bp_diastolic_values = np.clip(bp_diastolic_values, 40, 140)
    temperature_values = np.clip(temperature_values, 35, 41)
    oxygen_values = np.clip(oxygen_values, 70, 100)
    respiratory_values = np.clip(respiratory_values, 8, 40)
    glucose_values = np.clip(glucose_values, 40, 300)
    
    # Create the DataFrame
    data = {
        'timestamp': timestamps,
        'heart_rate': heart_rate_values.astype(int),
        'blood_pressure_systolic': bp_systolic_values.astype(int),
        'blood_pressure_diastolic': bp_diastolic_values.astype(int),
        'temperature': temperature_values.round(1),
        'oxygen_saturation': oxygen_values.astype(int),
        'respiratory_rate': respiratory_values.round(1),
        'glucose': glucose_values.astype(int)
    }
    
    return pd.DataFrame(data)

def process_vital_data(vital_data):
    """
    Process and analyze vital sign data for patient monitoring
    
    Args:
        vital_data (pd.DataFrame): DataFrame with vital sign measurements
        
    Returns:
        dict: Dictionary with analysis results
    """
    if vital_data.empty:
        return {"status": "error", "message": "No data available"}
    
    results = {
        "status": "success",
        "vital_stats": {},
        "summary": {}
    }
    
    # Calculate statistics for each vital sign
    for column in vital_data.columns:
        if column != 'timestamp':
            results["vital_stats"][column] = {
                "current": vital_data[column].iloc[-1],
                "min": vital_data[column].min(),
                "max": vital_data[column].max(),
                "mean": vital_data[column].mean(),
                "median": vital_data[column].median(),
                "std": vital_data[column].std()
            }
    
    # Generate summary based on latest readings
    latest = vital_data.iloc[-1]
    
    # Heart rate assessment
    hr = latest['heart_rate']
    if hr < 60:
        hr_status = "Bradycardia (low heart rate)"
    elif hr > 100:
        hr_status = "Tachycardia (high heart rate)"
    else:
        hr_status = "Normal"
    
    # Blood pressure assessment
    sys_bp = latest['blood_pressure_systolic']
    dia_bp = latest['blood_pressure_diastolic']
    
    if sys_bp < 90 or dia_bp < 60:
        bp_status = "Hypotension (low blood pressure)"
    elif sys_bp >= 140 or dia_bp >= 90:
        bp_status = "Hypertension (high blood pressure)"
    elif sys_bp >= 120 or dia_bp >= 80:
        bp_status = "Prehypertension"
    else:
        bp_status = "Normal"
    
    # Oxygen saturation
    oxygen = latest['oxygen_saturation']
    if oxygen < 90:
        oxygen_status = "Severe hypoxemia (critically low)"
    elif oxygen < 95:
        oxygen_status = "Moderate hypoxemia (low)"
    else:
        oxygen_status = "Normal"
    
    # Temperature
    temp = latest['temperature']
    if temp > 38.0:
        temp_status = "Fever"
    elif temp < 36.0:
        temp_status = "Hypothermia"
    else:
        temp_status = "Normal"
    
    # Add the assessments to the summary
    results["summary"] = {
        "heart_rate": hr_status,
        "blood_pressure": bp_status,
        "oxygen_saturation": oxygen_status,
        "temperature": temp_status
    }
    
    return results
