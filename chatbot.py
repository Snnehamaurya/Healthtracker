import streamlit as st
import pandas as pd
import re
from utils.chatbot import get_health_assistant_response, analyze_symptoms

def show_chatbot(patient_id):
    """
    Display AI-powered chatbot for health assistance
    
    Args:
        patient_id (str): ID of the selected patient
    """
    st.title("Health Assistant Chatbot")
    
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "error" not in st.session_state:
        st.session_state.error = None
    
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = None
    
    # Get patient info 
    patient_info = st.session_state.patients[patient_id]
    patient_data = st.session_state.patient_data[patient_id]
    
    # Patient context information
    patient_context = f"""
    Patient: {patient_info['name']}
    Age: {patient_info['age']}
    Gender: {patient_info['gender']}
    Medical Condition: {patient_info['condition']}
    """
    
    # Add vital signs summary to context if available
    if not patient_data.empty:
        latest = patient_data.iloc[-1]
        patient_context += f"""
        Latest Vital Signs:
        - Heart Rate: {int(latest['heart_rate'])} bpm
        - Blood Pressure: {int(latest['blood_pressure_systolic'])}/{int(latest['blood_pressure_diastolic'])} mmHg
        - Oxygen Saturation: {int(latest['oxygen_saturation'])}%
        - Temperature: {round(latest['temperature'], 1)}°C
        """
    
    # Display information about the chatbot
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.expander("About the Health Assistant", expanded=False):
            st.info("""
            This AI-powered health assistant can:
            
            - Answer general health questions
            - Interpret vital signs and health metrics
            - Provide information about medical conditions
            - Analyze symptoms and suggest possible conditions
            - Offer wellness and preventive health recommendations
            
            **Note:** This assistant does not provide medical diagnoses or replace professional medical advice.
            Always consult with healthcare professionals for medical concerns.
            """)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; border: 1px solid #ccc; padding: 15px; border-radius: 5px;">
        <h3>Symptom Analysis</h3>
        <p>Ask about specific symptoms to get health information</p>
        </div>
        """, unsafe_allow_html=True)
    
    # No OpenAI API key needed, we're using built-in medical knowledge
    
    # Display debug info if available
    if st.session_state.debug_info and st.session_state.user_role == "admin":
        st.code(st.session_state.debug_info, language="text")
    
    # Display error if there was one
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask a health-related question...", disabled=st.session_state.processing)
    
    # Show processing indicator
    if st.session_state.processing:
        with st.chat_message("assistant"):
            st.write("Thinking...")
    
    if prompt:
        # Display user message
        st.chat_message("user").write(prompt)
        
        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Check if the prompt is asking about symptoms
        symptoms_prompt = False
        symptom_patterns = [
            r"i.+(?:have|experiencing|suffering from)\b.+symptoms?",
            r"what.+(?:wrong with me)",
            r"(?:symptoms of|signs of)",
            r"i(?:'m| am) (?:feeling|having)",
            r"(?:diagnose|diagnosis)",
            r"why (?:do|am) I (?:have|feel)",
        ]
        
        for pattern in symptom_patterns:
            if re.search(pattern, prompt.lower()):
                symptoms_prompt = True
                break
        
        # Parse symptoms from the query if it seems to be symptom-related
        if symptoms_prompt:
            # Try to extract symptoms using a simple pattern
            potential_symptoms = re.findall(r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b(?:\s*,\s*|\s+and\s+|\s+with\s+|\s+plus\s+)', prompt.lower())
            
            # If symptoms detected and more than 2, use the symptom analyzer
            if len(potential_symptoms) >= 2:
                with st.spinner("Analyzing symptoms..."):
                    response = analyze_symptoms(potential_symptoms, patient_context)
            else:
                # Otherwise just use the normal response
                with st.spinner("Getting response..."):
                    response = get_health_assistant_response(prompt, patient_context)
        else:
            # Regular chatbot response
            with st.spinner("Getting response..."):
                response = get_health_assistant_response(prompt, patient_context)
        
        # Display assistant response
        st.chat_message("assistant").write(response)
        
        # Add to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Buttons for chat management
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("Test Assistant"):
            with st.spinner("Testing health assistant..."):
                test_response = get_health_assistant_response("Hello, can you tell me about the common cold?")
                st.success(f"Health assistant working! Response: {test_response[:50]}...")
                
    # Sample questions to help user get started
    st.markdown("---")
    
    # Two columns of sample questions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("General Health Questions")
        
        general_questions = [
            "What does a heart rate of 120 bpm indicate?",
            "What are normal oxygen saturation levels?",
            "How can I help manage diabetes?",
            "What lifestyle changes help with heart disease?",
            "How does hypertension affect the body?"
        ]
        
        for q in general_questions:
            if st.button(q, key=f"gen_{q}"):
                # Simulate clicking the chat input
                st.session_state.chat_history.append({"role": "user", "content": q})
                
                # Get response from AI
                with st.spinner("Getting response..."):
                    response = get_health_assistant_response(q, patient_context)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    with col2:
        st.subheader("Symptom Analysis Questions")
        
        symptom_questions = [
            "I have a headache and fever, what could it be?",
            "What causes chest pain and shortness of breath?",
            "I'm experiencing dizziness and nausea, why?",
            "What conditions cause fatigue and joint pain?",
            "I have a cough and sore throat, what should I do?"
        ]
        
        for q in symptom_questions:
            if st.button(q, key=f"sym_{q}"):
                # Simulate clicking the chat input
                st.session_state.chat_history.append({"role": "user", "content": q})
                
                # Get response from AI with symptom analysis
                with st.spinner("Analyzing symptoms..."):
                    # Extract potential symptoms
                    if "headache and fever" in q:
                        symptoms = ["headache", "fever"]
                    elif "chest pain and shortness of breath" in q:
                        symptoms = ["chest pain", "shortness of breath"]
                    elif "dizziness and nausea" in q:
                        symptoms = ["dizziness", "nausea"]
                    elif "fatigue and joint pain" in q:
                        symptoms = ["fatigue", "joint pain"]
                    elif "cough and sore throat" in q:
                        symptoms = ["cough", "sore throat"]
                    else:
                        symptoms = []
                    
                    if symptoms:
                        response = analyze_symptoms(symptoms, patient_context)
                    else:
                        response = get_health_assistant_response(q, patient_context)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()