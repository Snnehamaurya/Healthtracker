import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models.anomaly_detection import detect_anomalies, analyze_trends

def show_analytics(patient_id):
    """
    Display AI-powered analytics for patient health data
    
    Args:
        patient_id (str): ID of the selected patient
    """
    st.title("AI Health Analytics")
    
    # Get patient info and data
    patient_info = st.session_state.patients[patient_id]
    patient_data = st.session_state.patient_data[patient_id]
    
    st.header(f"AI Analysis for {patient_info['name']}")
    
    # Add tabs for different types of analysis
    tabs = st.tabs(["Trend Analysis", "Anomaly Detection", "Pattern Recognition", "Health Predictions"])
    
    # Tab 1: Trend Analysis
    with tabs[0]:
        st.subheader("Health Trend Analysis")
        
        # Select vital sign for trend analysis
        vital_sign = st.selectbox(
            "Select Vital Sign for Analysis",
            options=[
                "heart_rate", 
                "blood_pressure_systolic", 
                "blood_pressure_diastolic", 
                "temperature", 
                "oxygen_saturation", 
                "respiratory_rate",
                "glucose"
            ],
            key="trend_vital_sign"
        )
        
        # Get trend analysis for the selected vital sign
        trend_analysis = analyze_trends(patient_data, vital_sign)
        
        if trend_analysis["status"] == "success":
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Value", 
                         f"{trend_analysis['current_value']:.1f}" if vital_sign == 'temperature' 
                         else f"{int(trend_analysis['current_value'])}")
            
            with col2:
                st.metric("Average", 
                         f"{trend_analysis['mean']:.1f}" if vital_sign == 'temperature' 
                         else f"{int(trend_analysis['mean'])}")
            
            with col3:
                st.metric("Variability", 
                         f"{trend_analysis['std']:.2f}")
            
            # Display trend direction with indicators
            if trend_analysis["trend"] == "increasing":
                st.info("📈 This vital sign is **increasing** over the analysis period.")
            elif trend_analysis["trend"] == "decreasing":
                st.info("📉 This vital sign is **decreasing** over the analysis period.")
            else:
                st.info("📊 This vital sign is **stable** over the analysis period.")
            
            # Volatility analysis
            if trend_analysis["volatility"] == "high":
                st.warning("⚠️ The readings show **high volatility**, which might indicate instability.")
            elif trend_analysis["volatility"] == "low":
                st.success("✅ The readings show **low volatility**, indicating stability.")
            else:
                st.info("ℹ️ The readings show **normal volatility**.")
            
            # Visualization with trendline
            st.subheader("Trend Visualization")
            
            # Create a figure with both the actual data and a trendline
            fig = go.Figure()
            
            # Add the actual data
            fig.add_trace(go.Scatter(
                x=patient_data['timestamp'],
                y=patient_data[vital_sign],
                mode='lines',
                name='Actual Readings',
                line=dict(color='royalblue')
            ))
            
            # Add a rolling average trendline
            window_size = min(7, len(patient_data) // 4) if len(patient_data) > 4 else 1
            rolling_mean = patient_data[vital_sign].rolling(window=window_size).mean()
            
            fig.add_trace(go.Scatter(
                x=patient_data['timestamp'],
                y=rolling_mean,
                mode='lines',
                name=f'{window_size}-point Rolling Average',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f"{vital_sign.replace('_', ' ').title()} Trend Analysis",
                xaxis_title="Time",
                yaxis_title=vital_sign.replace('_', ' ').title(),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical interpretation based on vital sign
            st.subheader("Clinical Interpretation")
            
            if vital_sign == "heart_rate":
                if trend_analysis["trend"] == "increasing" and trend_analysis["current_value"] > 100:
                    st.warning("The increasing heart rate trend with a current elevated reading may indicate stress, anxiety, or potential cardiac issues.")
                elif trend_analysis["trend"] == "decreasing" and trend_analysis["current_value"] < 60:
                    st.warning("The decreasing heart rate trend with a current low reading may indicate bradycardia which should be monitored.")
                else:
                    st.success("The heart rate trend appears to be within normal parameters.")
            
            elif vital_sign == "blood_pressure_systolic":
                if trend_analysis["trend"] == "increasing" and trend_analysis["current_value"] > 140:
                    st.warning("The increasing systolic blood pressure trend with a current high reading may indicate hypertension.")
                elif trend_analysis["trend"] == "decreasing" and trend_analysis["current_value"] < 90:
                    st.warning("The decreasing systolic blood pressure trend with a current low reading may indicate hypotension.")
                else:
                    st.success("The systolic blood pressure trend appears to be within normal parameters.")
            
            elif vital_sign == "oxygen_saturation":
                if trend_analysis["trend"] == "decreasing" and trend_analysis["current_value"] < 95:
                    st.warning("The decreasing oxygen saturation trend with a current low reading may indicate respiratory issues.")
                else:
                    st.success("The oxygen saturation trend appears to be within normal parameters.")
            
            elif vital_sign == "temperature":
                if trend_analysis["trend"] == "increasing" and trend_analysis["current_value"] > 38.0:
                    st.warning("The increasing temperature trend with a current elevated reading may indicate infection or inflammation.")
                elif trend_analysis["trend"] == "decreasing" and trend_analysis["current_value"] < 36.0:
                    st.warning("The decreasing temperature trend with a current low reading may indicate hypothermia.")
                else:
                    st.success("The temperature trend appears to be within normal parameters.")
            
            elif vital_sign == "glucose":
                if trend_analysis["trend"] == "increasing" and trend_analysis["current_value"] > 140:
                    st.warning("The increasing glucose trend with a current high reading may indicate poor glycemic control.")
                elif trend_analysis["trend"] == "decreasing" and trend_analysis["current_value"] < 70:
                    st.warning("The decreasing glucose trend with a current low reading may indicate hypoglycemia risk.")
                else:
                    st.success("The glucose trend appears to be within normal parameters.")
        
        else:
            st.info("Insufficient data for trend analysis. Please select a different vital sign or check back later.")
    
    # Tab 2: Anomaly Detection
    with tabs[1]:
        st.subheader("Anomaly Detection")
        
        # Information about the anomaly detection approach
        st.info("""
        This analysis uses machine learning to identify measurements that fall outside the expected patterns.
        Anomalies could indicate potential health concerns or measurement errors.
        """)
        
        # Get anomaly detection results for all vital signs
        all_anomalies = detect_anomalies(
            patient_data.iloc[-1:],  # Latest data point
            patient_data.iloc[:-1]   # Historical data
        )
        
        # Display anomalies by vital sign
        vital_features = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'temperature', 'oxygen_saturation', 'respiratory_rate'
        ]
        
        # Count the anomalies
        anomaly_count = sum(1 for feature, is_anomaly in all_anomalies.items() if is_anomaly)
        
        # Display summary
        if anomaly_count > 0:
            st.warning(f"Detected {anomaly_count} anomalies in the latest readings!")
        else:
            st.success("No anomalies detected in the latest readings.")
        
        # Create visualization of normal vs anomalous readings
        st.subheader("Anomaly Visualization")
        
        # Create a more advanced anomaly visualization
        # Run anomaly detection on all data points
        anomaly_results = []
        
        # For each data point, check if it's an anomaly compared to previous points
        for i in range(1, len(patient_data)):
            point_anomalies = detect_anomalies(
                patient_data.iloc[i:i+1],
                patient_data.iloc[:i]
            )
            
            is_anomaly = any(point_anomalies.values())
            anomaly_results.append(is_anomaly)
        
        # Add first point (can't be an anomaly with no history)
        anomaly_results.insert(0, False)
        
        # Add anomaly status to data
        patient_data_with_anomalies = patient_data.copy()
        patient_data_with_anomalies['is_anomaly'] = anomaly_results
        
        # Let user select which vital sign to visualize
        anomaly_vital_sign = st.selectbox(
            "Select Vital Sign for Anomaly Visualization",
            options=vital_features,
            key="anomaly_vital_sign"
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add normal readings
        normal_data = patient_data_with_anomalies[~patient_data_with_anomalies['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data[anomaly_vital_sign],
            mode='markers+lines',
            name='Normal Readings',
            marker=dict(color='blue', size=8)
        ))
        
        # Add anomalous readings
        anomaly_data = patient_data_with_anomalies[patient_data_with_anomalies['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomaly_data['timestamp'],
            y=anomaly_data[anomaly_vital_sign],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=12, symbol='circle-open')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Anomaly Detection for {anomaly_vital_sign.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=anomaly_vital_sign.replace('_', ' ').title(),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display details of the latest anomalies
        if anomaly_count > 0:
            st.subheader("Anomaly Details")
            
            for feature, is_anomaly in all_anomalies.items():
                if is_anomaly:
                    current_value = patient_data[feature].iloc[-1]
                    mean_value = patient_data[feature].iloc[:-1].mean()
                    std_value = patient_data[feature].iloc[:-1].std()
                    z_score = (current_value - mean_value) / std_value if std_value > 0 else 0
                    
                    st.warning(f"**{feature.replace('_', ' ').title()}**: Current value ({current_value:.1f}) deviates significantly from the expected range.")
                    st.write(f"Expected range: {mean_value - 2*std_value:.1f} to {mean_value + 2*std_value:.1f} (based on historical data)")
                    st.write(f"Deviation: {z_score:.2f} standard deviations from the mean")
        
    # Tab 3: Pattern Recognition
    with tabs[2]:
        st.subheader("Pattern Recognition")
        
        st.info("""
        This analysis identifies patterns and relationships between different vital signs.
        Understanding these relationships can provide insights into the patient's overall health.
        """)
        
        # Correlation Analysis
        st.subheader("Vital Signs Correlation")
        
        # Create correlation matrix
        vital_columns = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'temperature', 'oxygen_saturation', 'respiratory_rate', 'glucose'
        ]
        
        correlation = patient_data[vital_columns].corr()
        
        # Plot correlation heatmap
        fig = px.imshow(
            correlation,
            labels=dict(color="Correlation"),
            x=[col.replace('_', ' ').title() for col in vital_columns],
            y=[col.replace('_', ' ').title() for col in vital_columns],
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        
        fig.update_layout(
            height=500,
            title="Correlation Between Vital Signs"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpret the correlations
        st.subheader("Correlation Interpretation")
        
        # Find the strongest positive and negative correlations
        mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
        tri_correlation = correlation.mask(~mask)
        
        # Get top positive correlation
        top_pos_corr = tri_correlation.unstack().dropna().sort_values(ascending=False).head(3)
        
        # Get top negative correlation
        top_neg_corr = tri_correlation.unstack().dropna().sort_values().head(3)
        
        # Display the results
        if not top_pos_corr.empty:
            st.write("**Strongest Positive Correlations:**")
            for (var1, var2), value in top_pos_corr.items():
                st.write(f"- {var1.replace('_', ' ').title()} and {var2.replace('_', ' ').title()}: {value:.2f}")
                
                # Interpretation based on the specific correlation
                if var1 == 'heart_rate' and var2 == 'respiratory_rate' or var2 == 'heart_rate' and var1 == 'respiratory_rate':
                    st.write("  💡 *Heart rate and respiratory rate often increase together during physical activity or stress.*")
                elif (var1 == 'blood_pressure_systolic' and var2 == 'blood_pressure_diastolic' or 
                      var2 == 'blood_pressure_systolic' and var1 == 'blood_pressure_diastolic'):
                    st.write("  💡 *Systolic and diastolic blood pressure typically rise and fall together.*")
        
        if not top_neg_corr.empty:
            st.write("**Strongest Negative Correlations:**")
            for (var1, var2), value in top_neg_corr.items():
                st.write(f"- {var1.replace('_', ' ').title()} and {var2.replace('_', ' ').title()}: {value:.2f}")
                
                # Interpretation based on the specific correlation
                if var1 == 'heart_rate' and var2 == 'oxygen_saturation' or var2 == 'heart_rate' and var1 == 'oxygen_saturation':
                    st.write("  💡 *A negative correlation between heart rate and oxygen saturation can indicate that the heart is working harder to compensate for lower oxygen levels.*")
        
        # Pattern Clustering Analysis
        st.subheader("Vital Signs Clustering")
        
        # Prepare data for clustering
        X = patient_data[vital_columns].copy()
        
        # Handle missing values if any
        X = X.fillna(X.mean())
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply K-Means clustering
        n_clusters = min(3, len(X))  # Choose a reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create a DataFrame for visualization
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': clusters,
            'Timestamp': patient_data['timestamp']
        })
        
        # Get cluster centers
        centers_pca = pca.transform(scaler.transform(kmeans.cluster_centers_))
        
        # Create scatter plot
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=['Timestamp'],
            title='Clustering of Vital Signs Patterns',
        )
        
        # Add cluster centers
        for i, (x, y) in enumerate(centers_pca):
            fig.add_annotation(
                x=x, y=y,
                text=f"Cluster {i}",
                showarrow=True,
                arrowhead=1
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpret the clusters
        st.write("**Cluster Interpretation:**")
        st.write("Each point represents a set of vital signs at a specific time. Points clustered together show similar health patterns.")
        
        # Calculate cluster characteristics
        cluster_means = []
        for i in range(n_clusters):
            cluster_data = X[clusters == i]
            if not cluster_data.empty:
                cluster_mean = cluster_data.mean()
                cluster_means.append((i, cluster_mean))
        
        # Display cluster characteristics
        for i, means in cluster_means:
            st.write(f"**Cluster {i} Characteristics:**")
            for col, mean in means.items():
                feature_name = col.replace('_', ' ').title()
                overall_mean = X[col].mean()
                
                # Determine if the value is high, low or normal compared to overall
                if mean > overall_mean * 1.1:
                    status = "High"
                    emoji = "⬆️"
                elif mean < overall_mean * 0.9:
                    status = "Low"
                    emoji = "⬇️"
                else:
                    status = "Normal"
                    emoji = "➡️"
                
                st.write(f"- {feature_name}: {mean:.1f} ({emoji} {status})")
    
    # Tab 4: Health Predictions
    with tabs[3]:
        st.subheader("Health Predictions")
        
        st.info("""
        This analysis uses trend data to make predictions about future health metrics.
        These predictions are based on statistical models and should be used as guidance only.
        """)
        
        # Select vital sign for prediction
        prediction_vital = st.selectbox(
            "Select Vital Sign for Prediction",
            options=[
                "heart_rate", 
                "blood_pressure_systolic", 
                "blood_pressure_diastolic", 
                "temperature", 
                "oxygen_saturation", 
                "respiratory_rate",
                "glucose"
            ],
            key="prediction_vital_sign"
        )
        
        # Time period for prediction
        prediction_hours = st.slider("Prediction Hours Ahead", 6, 72, 24)
        
        # Check if we have enough data
        if len(patient_data) < 24:
            st.warning("Not enough historical data for reliable predictions. Need at least 24 hours of data.")
        else:
            # Get the data for the selected vital sign
            y = patient_data[prediction_vital].values
            
            # Simple moving average model for prediction
            window_size = min(24, len(y) // 2)
            
            # Calculate the trend
            trend = np.mean(y[-window_size:] - y[-2*window_size:-window_size]) / window_size
            
            # Calculate the last average
            last_avg = np.mean(y[-window_size:])
            
            # Generate prediction times
            last_time = patient_data['timestamp'].iloc[-1]
            future_times = pd.date_range(start=last_time, periods=prediction_hours+1, freq='H')[1:]
            
            # Generate predictions
            predictions = [last_avg + trend * i for i in range(1, prediction_hours+1)]
            
            # Create prediction ranges (uncertainty increases with time)
            lower_bounds = []
            upper_bounds = []
            
            for i in range(prediction_hours):
                # Increasing uncertainty with time
                uncertainty = np.std(y[-window_size:]) * (1 + i/window_size)
                lower_bounds.append(predictions[i] - uncertainty)
                upper_bounds.append(predictions[i] + uncertainty)
            
            # Create visualization
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=patient_data['timestamp'],
                y=patient_data[prediction_vital],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add prediction
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines',
                name='Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            # Add prediction confidence interval
            fig.add_trace(go.Scatter(
                x=list(future_times) + list(future_times)[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{prediction_vital.replace('_', ' ').title()} Prediction for Next {prediction_hours} Hours",
                xaxis_title="Time",
                yaxis_title=prediction_vital.replace('_', ' ').title(),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical interpretation
            st.subheader("Prediction Interpretation")
            
            # Get last value and prediction endpoint
            last_value = y[-1]
            predicted_end = predictions[-1]
            
            # Calculate percent change
            percent_change = ((predicted_end - last_value) / last_value) * 100
            
            if abs(percent_change) < 5:
                st.success(f"Prediction suggests stable {prediction_vital.replace('_', ' ')} levels over the next {prediction_hours} hours.")
            elif percent_change > 5:
                st.warning(f"Prediction suggests an increasing trend in {prediction_vital.replace('_', ' ')} by approximately {percent_change:.1f}% over the next {prediction_hours} hours.")
            else:
                st.warning(f"Prediction suggests a decreasing trend in {prediction_vital.replace('_', ' ')} by approximately {abs(percent_change):.1f}% over the next {prediction_hours} hours.")
            
            # Add specific clinical notes based on the vital sign
            if prediction_vital == "heart_rate":
                if predicted_end > 100:
                    st.warning("The predicted elevated heart rate may indicate increased stress or potential cardiac issues.")
                elif predicted_end < 60:
                    st.warning("The predicted low heart rate may indicate bradycardia which should be monitored.")
            
            elif prediction_vital == "blood_pressure_systolic":
                if predicted_end > 140:
                    st.warning("The predicted elevated systolic blood pressure may indicate hypertension risk.")
                elif predicted_end < 90:
                    st.warning("The predicted low systolic blood pressure may indicate hypotension risk.")
            
            elif prediction_vital == "oxygen_saturation":
                if predicted_end < 95:
                    st.warning("The predicted decreased oxygen saturation may indicate potential respiratory issues.")
            
            # Disclaimer
            st.info("Note: These predictions are based on statistical modeling and should be used as a guide only. Medical decisions should be made by healthcare professionals.")
