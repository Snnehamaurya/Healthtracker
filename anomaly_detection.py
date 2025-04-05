import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(new_data, historical_data, contamination=0.05):
    """
    Detect anomalies in new patient data based on historical patterns
    
    Args:
        new_data (pd.DataFrame): Latest vital sign measurements
        historical_data (pd.DataFrame): Historical vital sign measurements
        contamination (float): Expected proportion of outliers in the data
        
    Returns:
        dict: Dictionary with feature names as keys and boolean values indicating anomalies
    """
    # Features to monitor for anomalies
    vital_features = ['heart_rate', 'blood_pressure_systolic', 
                     'blood_pressure_diastolic', 'temperature', 
                     'oxygen_saturation', 'respiratory_rate']
    
    anomalies = {}
    
    # Process each vital sign separately
    for feature in vital_features:
        if feature in historical_data.columns and feature in new_data.columns:
            # Get historical data for this feature
            feature_data = historical_data[feature].values.reshape(-1, 1)
            
            # Skip if not enough historical data
            if len(feature_data) < 10:
                anomalies[feature] = False
                continue
                
            # Standardize the data
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            
            # Train isolation forest model
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(feature_data_scaled)
            
            # Prepare new data point
            new_point = scaler.transform(new_data[feature].values.reshape(-1, 1))
            
            # Predict if it's an anomaly (-1 for anomaly, 1 for normal)
            prediction = model.predict(new_point)[0]
            anomalies[feature] = (prediction == -1)
    
    return anomalies

def analyze_trends(patient_data, feature, window_size=7):
    """
    Analyze trends in patient vital signs over time
    
    Args:
        patient_data (pd.DataFrame): Historical vital sign measurements
        feature (str): Feature to analyze
        window_size (int): Rolling window size for trend analysis
        
    Returns:
        dict: Dictionary with trend analysis results
    """
    if feature not in patient_data.columns or len(patient_data) < window_size:
        return {"status": "insufficient_data"}
    
    # Calculate rolling statistics
    rolling_mean = patient_data[feature].rolling(window=window_size).mean()
    rolling_std = patient_data[feature].rolling(window=window_size).std()
    
    # Calculate the trend direction
    latest_mean = rolling_mean.iloc[-1]
    previous_mean = rolling_mean.iloc[-window_size] if len(rolling_mean) >= window_size else None
    
    trend_direction = "stable"
    if previous_mean is not None:
        percent_change = ((latest_mean - previous_mean) / previous_mean) * 100
        if percent_change > 5:
            trend_direction = "increasing"
        elif percent_change < -5:
            trend_direction = "decreasing"
    
    # Determine volatility
    latest_std = rolling_std.iloc[-1]
    volatility = "normal"
    if latest_std > 1.5 * rolling_std.mean():
        volatility = "high"
    elif latest_std < 0.5 * rolling_std.mean():
        volatility = "low"
    
    return {
        "status": "success",
        "trend": trend_direction,
        "volatility": volatility,
        "current_value": patient_data[feature].iloc[-1],
        "mean": latest_mean,
        "std": latest_std
    }
