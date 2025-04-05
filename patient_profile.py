import streamlit as st
import pandas as pd

def show_patient_profile(patient_id):
    """
    Display and manage patient profile information
    
    Args:
        patient_id (str): ID of the selected patient
    """
    st.title("Patient Profile Management")
    
    # Get current patient info
    patient_info = st.session_state.patients[patient_id]
    
    # Display current information and allow editing
    st.subheader(f"Profile: {patient_info['name']} ({patient_id})")
    
    # Create form for editing
    with st.form("edit_profile"):
        # Split form into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value=patient_info["name"])
            age = st.number_input("Age", 
                                 min_value=0, 
                                 max_value=120, 
                                 value=patient_info["age"],
                                 step=1)
            gender = st.selectbox("Gender", 
                                 options=["Male", "Female", "Other"], 
                                 index=["Male", "Female", "Other"].index(patient_info["gender"]))
        
        with col2:
            condition = st.text_input("Primary Medical Condition", 
                                     value=patient_info["condition"])
            
            # Additional fields
            blood_type = st.selectbox("Blood Type", 
                                     options=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                     index=0)
            
            allergies = st.text_area("Allergies", 
                                     value="None" if "allergies" not in patient_info else patient_info["allergies"])
        
        # Add notes field
        medical_notes = st.text_area("Medical Notes", 
                                   value="" if "medical_notes" not in patient_info else patient_info["medical_notes"],
                                   height=150)
        
        # Submit button for the form
        submitted = st.form_submit_button("Update Profile")
        
        if submitted:
            # Update patient information
            st.session_state.patients[patient_id] = {
                "name": name,
                "age": age,
                "gender": gender,
                "condition": condition,
                "blood_type": blood_type,
                "allergies": allergies,
                "medical_notes": medical_notes
            }
            
            st.success("Profile updated successfully!")
            
    # Display health metrics summary
    st.markdown("---")
    st.subheader("Health Metrics Summary")
    
    patient_data = st.session_state.patient_data[patient_id]
    
    if not patient_data.empty:
        # Get metrics from the most recent data point
        latest = patient_data.iloc[-1]
        
        # Calculate averages from recent data (last 24 points)
        recent = patient_data.iloc[-24:]
        
        metrics_cols = st.columns(3)
        
        # Heart rate metrics
        with metrics_cols[0]:
            st.metric(
                label="Heart Rate", 
                value=f"{int(latest['heart_rate'])} bpm", 
                delta=f"{int(latest['heart_rate'] - recent['heart_rate'].mean())}"
            )
            
            st.metric(
                label="Systolic BP", 
                value=f"{int(latest['blood_pressure_systolic'])} mmHg", 
                delta=f"{int(latest['blood_pressure_systolic'] - recent['blood_pressure_systolic'].mean())}"
            )
            
            st.metric(
                label="Diastolic BP", 
                value=f"{int(latest['blood_pressure_diastolic'])} mmHg", 
                delta=f"{int(latest['blood_pressure_diastolic'] - recent['blood_pressure_diastolic'].mean())}"
            )
        
        # Oxygen and temperature metrics
        with metrics_cols[1]:
            st.metric(
                label="Oxygen Saturation", 
                value=f"{int(latest['oxygen_saturation'])}%", 
                delta=f"{int(latest['oxygen_saturation'] - recent['oxygen_saturation'].mean())}"
            )
            
            st.metric(
                label="Temperature", 
                value=f"{latest['temperature']:.1f}°C", 
                delta=f"{(latest['temperature'] - recent['temperature'].mean()):.1f}"
            )
            
            st.metric(
                label="Respiratory Rate", 
                value=f"{latest['respiratory_rate']:.1f} BrPM", 
                delta=f"{(latest['respiratory_rate'] - recent['respiratory_rate'].mean()):.1f}"
            )
        
        # Glucose metrics
        with metrics_cols[2]:
            st.metric(
                label="Glucose Level", 
                value=f"{int(latest['glucose'])} mg/dL", 
                delta=f"{int(latest['glucose'] - recent['glucose'].mean())}"
            )
    else:
        st.info("No health metrics data available for this patient.")
    
    # Patient history section
    st.markdown("---")
    st.subheader("Patient History")
    
    # Create expandable sections for different aspects of patient history
    with st.expander("Medical History", expanded=True):
        if "medical_history" in patient_info:
            st.write(patient_info["medical_history"])
        else:
            # Example medical history form
            with st.form("medical_history_form"):
                chronic_conditions = st.multiselect(
                    "Chronic Conditions",
                    options=["Hypertension", "Diabetes", "Asthma", "Heart Disease", "COPD", "Cancer", "None"],
                    default=["None"] if patient_info["condition"] == "Normal" else [patient_info["condition"]]
                )
                
                previous_surgeries = st.text_area("Previous Surgeries", value="None")
                
                family_history = st.text_area("Family Medical History", value="None")
                
                smoking = st.radio("Smoking Status", ["Never", "Former", "Current"], index=0)
                
                alcohol = st.radio("Alcohol Consumption", ["None", "Occasional", "Moderate", "Heavy"], index=0)
                
                save_history = st.form_submit_button("Save Medical History")
                
                if save_history:
                    medical_history = {
                        "chronic_conditions": chronic_conditions,
                        "previous_surgeries": previous_surgeries,
                        "family_history": family_history,
                        "smoking": smoking,
                        "alcohol": alcohol
                    }
                    
                    # Update patient info with medical history
                    patient_info["medical_history"] = medical_history
                    st.session_state.patients[patient_id] = patient_info
                    
                    st.success("Medical history saved successfully!")
    
    with st.expander("Medications", expanded=False):
        if "medications" in patient_info:
            # Display existing medications
            for med in patient_info["medications"]:
                st.write(f"**{med['name']}**: {med['dosage']} - {med['frequency']}")
        
        # Form to add new medication
        with st.form("add_medication"):
            st.subheader("Add Medication")
            
            med_name = st.text_input("Medication Name")
            med_dosage = st.text_input("Dosage")
            med_frequency = st.text_input("Frequency")
            
            add_med = st.form_submit_button("Add Medication")
            
            if add_med and med_name and med_dosage and med_frequency:
                new_med = {
                    "name": med_name,
                    "dosage": med_dosage,
                    "frequency": med_frequency
                }
                
                if "medications" not in patient_info:
                    patient_info["medications"] = []
                
                patient_info["medications"].append(new_med)
                st.session_state.patients[patient_id] = patient_info
                
                st.success(f"Added {med_name} to medications list!")
    
    # Emergency contacts
    with st.expander("Emergency Contacts", expanded=False):
        if "emergency_contacts" in patient_info:
            # Display existing contacts
            for contact in patient_info["emergency_contacts"]:
                st.write(f"**{contact['name']}**: {contact['relationship']} - {contact['phone']}")
        
        # Form to add new contact
        with st.form("add_contact"):
            st.subheader("Add Emergency Contact")
            
            contact_name = st.text_input("Contact Name")
            contact_relationship = st.text_input("Relationship")
            contact_phone = st.text_input("Phone Number")
            
            add_contact = st.form_submit_button("Add Contact")
            
            if add_contact and contact_name and contact_relationship and contact_phone:
                new_contact = {
                    "name": contact_name,
                    "relationship": contact_relationship,
                    "phone": contact_phone
                }
                
                if "emergency_contacts" not in patient_info:
                    patient_info["emergency_contacts"] = []
                
                patient_info["emergency_contacts"].append(new_contact)
                st.session_state.patients[patient_id] = patient_info
                
                st.success(f"Added {contact_name} to emergency contacts!")
