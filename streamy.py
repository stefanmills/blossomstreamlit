import streamlit as st
import hashlib
import numpy as np
import pickle
import requests
import io
import time
from streamlit_lottie import st_lottie

# Function to load Lottie animation from a URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations URLs
lottie_default = "https://lottie.host/0e5e5fd2-62d9-406e-9733-e393d8ae38c1/bYbgzbADHS.json"  # Example URL for default animation
lottie_no_default = "https://lottie.host/748445dc-0823-444f-8cd1-6629ccc7d42d/rEsovbxROq.json"  # Example URL for no default animation
lottie_home="https://lottie.host/ead6891f-c6ca-47e3-9711-bd352b01e645/HyXTnlK4NC.json"

# Load animations
default_animation = load_lottie_url(lottie_default)
no_default_animation = load_lottie_url(lottie_no_default)
home_page=load_lottie_url(lottie_home)

# Function to download file from Google Drive
def download_file_from_google_drive(file_id):
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(download_url)
    
    # Check if the response is valid
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download file from Google Drive. Status code: {response.status_code}")
        return None

# Google Drive file IDs
model_file_id = "1kaDZVrlYPkRDuAICLuNX2snJKt2lrchb"  # Model file ID

# Load the trained model and scaler from Google Drive
model_content = download_file_from_google_drive(model_file_id)
message_placeholder = st.empty()

# Your existing code to download the model file
if model_content:
    try:
        model = pickle.load(io.BytesIO(model_content))
        
        # Show loading spinner
        with message_placeholder:
            with st.spinner('Model is loading...'):
                time.sleep(3)  # Simulate the loading process with a 3-second delay
        
        # Show success message for 3 seconds
        message_placeholder.success("Model loaded successfully.")
        time.sleep(3)
        
        # Clear the message after 3 seconds
        message_placeholder.empty()

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()  # Stop the execution if the model fails to load
else:
    st.stop()  # Stop the execution if the model cannot be loaded

# Function to hash the password (though not used in this case)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

st_lottie(home_page, height=200, key="home")
# Title for the application
st.markdown("<h2 style='text-align: center;'>SPJ Savings and Loans</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>...Build Your Future With Us</h5>", unsafe_allow_html=True)

# Session state to track login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Check if the user is logged in
if not st.session_state['logged_in']:
    # Login Form
    st.text_input("Username", value="Admin", disabled=True)  # Make username uneditable

    password = st.text_input("Password", type="password")

    # Login button
    if st.button("LOGIN"):
        # Check if the password is correct (set to '1234')
        if password == "1234":
            st.session_state['logged_in'] = True
            st.success("Login successful! Redirecting to the model page...")
        else:
            st.error("Incorrect password. Please try again.")
else:
    # If logged in, display the predictive model page
    st.markdown("<h2 style='text-align: center;'>SPJ Savings and Loans Predictive Model</h2>", unsafe_allow_html=True)

    # Instructions
    st.write("Please fill in the details below")

    # Client Information
    st.markdown("<h2 style='text-align: center;'>Client Information</h2>", unsafe_allow_html=True)

    # Name of Client
    client_name = st.text_input("Name of Client")

    # Create 3 columns for Age, Annual Income, and Home Ownership Status
    col1, col2, col3 = st.columns(3)

    with col1:
        # Age input
        age = st.number_input("Age (18 and above)", min_value=18, value=18)

    with col2:
        # Annual income input
        annual_income = st.number_input("Annual Income (GH₵)", min_value=0.0, step=1000.0, value=0.0)

    with col3:
        # Home ownership dropdown options
        home_ownership = st.selectbox("Home Ownership Status", ("", "RENT", "OWN", "MORTGAGE", "OTHER"))

    # Create another row with 3 columns for Employment Duration, Loan Purpose, and Loan Amount
    col4, col5, col6 = st.columns(3)

    with col4:
        # Employment Duration
        employment_duration = st.number_input("Employment Duration (Years)", min_value=0, value=0)

    with col5:
        # Purpose of the loan dropdown options
        loan_purpose = st.selectbox("Purpose of the Loan", ("", "PERSONAL", "EDUCATION", "VENTURE", 
                                                            "HOMEIMPROVEMENT", "MEDICAL", "DEBTCONSOLIDATION"))

    with col6:
        # Loan applied
        loan_applied = st.number_input("Loan Applied (GH₵)", min_value=0.0, step=500.0, value=0.0)
    
    col7, col8, col9 = st.columns(3)
    with col7:
        rate = st.number_input("Rate (%)", min_value=0.0, step=0.1, value=0.0)
    with col8:
        default = st.selectbox("Default", ("", "Yes", "No"))
    with col9:
        credit_history = st.number_input("Credit History", min_value=0.0, step=1.0, value=0.0)
    
    # Parameter Definitions
    st.markdown("""
    ### Parameter Definitions

    - **Age**: Age of the loan applicant.
    - **Income (GH₵)**: Annual income of the loan applicant in cedis.
    - **Home**: Home ownership status of applicant.
    - **Employment Duration (Years)**: Employment length in years.
    - **Intent**: Purpose of the loan.
    - **Loan Applied (GH₵)**: Loan amount applied for in cedis.
    - **Rate**: Interest rate on the loan.
    - **Default**: Whether the applicant has defaulted on a loan previously.
    - **Credit History**: Length of the applicant's credit history.
    """)
    
    # Button for prediction
    if st.button("Predict"):
        # Preprocess the input data (like during model training)

        # 1. One-hot encode 'Home' column (4 categories: RENT, OWN, MORTGAGE, OTHER)
        home_onehot = [0, 0, 0, 0]  # Initialize as 4 zeros
        home_mapping = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
        if home_ownership != "":
            home_onehot[home_mapping[home_ownership]] = 1

        # 2. One-hot encode 'Intent' column (6 categories)
        intent_onehot = [0, 0, 0, 0, 0, 0]  # Initialize as 6 zeros
        intent_mapping = {"PERSONAL": 0, "EDUCATION": 1, "VENTURE": 2, 
                          "HOMEIMPROVEMENT": 3, "MEDICAL": 4, "DEBTCONSOLIDATION": 5}
        if loan_purpose != "":
            intent_onehot[intent_mapping[loan_purpose]] = 1

        # 3. Map default to numeric
        default_value = 1 if default == "Yes" else 0

        # 4. Construct the input array (needs to be in the same format as training)
        # Combine numerical features with one-hot encoded categorical features
        input_data = np.array([[annual_income, loan_applied, rate, credit_history, default_value, employment_duration] 
                               + home_onehot + intent_onehot])

        # 5. Scale the numerical features
        numerical_features_indices = [0, 1, 2, 3, 5]  # Adjust indices based on the input order
        #input_data[:, numerical_features_indices] = scaler.transform(input_data[:, numerical_features_indices])

        # 6. Make the prediction using the model
        prediction = model.predict(input_data)

        # 7. Output the result
        if prediction[0] == 0:
            st.error(f"The model predicts that {client_name} **WILL LIKELY DEFAULT**.")
            st_lottie(default_animation, height=300, key="default")
            st.write("""
            ### Disclaimer!!!
            This application uses a *predictive model* to provide insights based on the data you input. Please note that the predictions are based on historical data and various assumptions. While the model is designed to be as accurate as possible, *no prediction can be 100% accurate*.

            **We recommend using the predictions as guidance and supplementing them with additional research and analysis when making decisions.**
            """)
        else:
            st.success(f"The model predicts that {client_name} **WILL LIKELY NOT DEFAULT**.")
            st_lottie(no_default_animation, height=300, key="no_default")
            st.write("""
            ### Disclaimer!!!
            This application uses a *predictive model* to provide insights based on the data you input. Please note that the predictions are based on historical data and various assumptions. While the model is designed to be as accurate as possible, *no prediction can be 100% accurate*.

            **We recommend using the predictions as guidance and supplementing them with additional research and analysis when making decisions.**
            """)

    # Back button to log out and return to login page
    
    if st.button("Back to Login"):
        st.session_state['logged_in'] = False
    st.markdown("© SPJ Savings and Loans, 2024")
