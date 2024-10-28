import streamlit as st
import numpy as np
import pickle
import requests
import time
from streamlit_lottie import st_lottie

# Load Lottie animations from URLs
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations URLs
lottie_default = "https://lottie.host/0e5e5fd2-62d9-406e-9733-e393d8ae38c1/bYbgzbADHS.json"
lottie_no_default = "https://lottie.host/748445dc-0823-444f-8cd1-6629ccc7d42d/rEsovbxROq.json"
lottie_home = "https://lottie.host/ead6891f-c6ca-47e3-9711-bd352b01e645/HyXTnlK4NC.json"

# Load animations
default_animation = load_lottie_url(lottie_default)
no_default_animation = load_lottie_url(lottie_no_default)
home_page = load_lottie_url(lottie_home)

# Function to download model and scaler from a URL
def download_model(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download model. Status code: {response.status_code}")
        return None

# Google Drive URL for model and scaler (replace with the correct URL)
model_file_url = "https://drive.google.com/uc?id=1KzAn3T7RlCmgjnbBq1RwEW6KZXnZxzBg&export=download"

# Load the trained model and scaler
model_content = download_model(model_file_url)
message_placeholder = st.empty()

if model_content:
    try:
        # Load the model and scaler from the pickle file
        model, scaler = pickle.loads(model_content)
        
        with message_placeholder:
            with st.spinner('Model and scaler are loading...'):
                time.sleep(3)
        
        message_placeholder.success("Model and scaler loaded successfully.")
        time.sleep(3)
        message_placeholder.empty()

    except Exception as e:
        st.error(f"Error loading model and scaler: {e}")
        st.stop()
else:
    st.stop()

# Streamlit UI setup
st_lottie(home_page, height=200, key="home")
st.markdown("<h2 style='text-align: center;'>SPJ Savings and Loans</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>...Build Your Future With Us</h5>", unsafe_allow_html=True)

# Login mechanism
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.text_input("Username", value="Admin", disabled=True)
    password = st.text_input("Password", type="password")

    if st.button("LOGIN"):
        if password == "1234":
            st.session_state['logged_in'] = True
            st.success("Login successful! Redirecting to the model page...")
        else:
            st.error("Incorrect password. Please try again.")
else:
    st.markdown("<h2 style='text-align: center;'>SPJ Savings and Loans Predictive Model</h2>", unsafe_allow_html=True)
    st.write("Please fill in the details below")

    # Collect user inputs
    client_name = st.text_input("Name of Client")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (18 and above)", min_value=18, value=18)
    with col2:
        annual_income = st.number_input("Annual Income (GH₵)", min_value=0.0, step=1000.0, value=0.0)
    with col3:
        home_ownership = st.selectbox("Home Ownership Status", ("", "RENT", "OWN", "MORTGAGE", "OTHER"))

    col4, col5, col6 = st.columns(3)
    with col4:
        employment_duration = st.number_input("Employment Duration (Years)", min_value=0, value=0)
    with col5:
        loan_purpose = st.selectbox("Purpose of the Loan", ("", "PERSONAL", "EDUCATION", "VENTURE", 
                                                            "HOMEIMPROVEMENT", "MEDICAL", "DEBTCONSOLIDATION"))
    with col6:
        loan_applied = st.number_input("Loan Applied (GH₵)", min_value=0.0, step=500.0, value=0.0)

    col7, col8, col9 = st.columns(3)
    with col7:
        rate = st.number_input("Rate (%)", min_value=0.0, step=0.1, value=0.0)
    with col8:
        default = st.selectbox("Default", ("", "Yes", "No"))
    with col9:
        credit_history = st.number_input("Credit History", min_value=0.0, step=1.0, value=0.0)
    percentage_income = loan_applied / annual_income if annual_income > 0 else 0

    # One-hot encoding for home ownership
    if home_ownership == "RENT":
        home_onehot = [0, 0, 1]
    elif home_ownership == "OWN":
        home_onehot = [0, 1, 0]
    elif home_ownership == "MORTGAGE":
        home_onehot = [0, 0, 0]
    elif home_ownership == "OTHER":
        home_onehot = [1, 0, 0]
    else:
        home_onehot = [0, 0, 0]  # Default if no selection

    # One-hot encoding for loan purpose
    if loan_purpose == "PERSONAL":
        intent_onehot = [0, 0, 0, 1, 0]
    elif loan_purpose == "EDUCATION":
        intent_onehot = [1, 0, 0, 0, 0]
    elif loan_purpose == "VENTURE":
        intent_onehot = [0, 0, 0, 0, 1]
    elif loan_purpose == "HOMEIMPROVEMENT":
        intent_onehot = [0, 1, 0, 0, 0]
    elif loan_purpose == "MEDICAL":
        intent_onehot = [0, 0, 1, 0, 0]
    elif loan_purpose == "DEBTCONSOLIDATION":
        intent_onehot = [0, 0, 0, 0, 0]
    else:
        intent_onehot = [0, 0, 0, 0, 0]  # Default if no selection

    default_value = 1 if default == "Yes" else 0

    # Construct input array
    input_data = np.array([[age, annual_income, employment_duration, loan_applied, rate, percentage_income, default_value, credit_history] 
                           + home_onehot + intent_onehot])

    # Apply scaling to numerical features
    numerical_features_indices = [0, 1, 2, 3, 4, 5]
    input_data[:, numerical_features_indices] = scaler.transform(input_data[:, numerical_features_indices])

    # Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)
            if prediction[0] == 0:
                st.error(f"The model predicts that {client_name} **WILL LIKELY DEFAULT**.")
                st_lottie(default_animation, height=300, key="default")
            else:
                st.success(f"The model predicts that {client_name} **WILL LIKELY NOT DEFAULT**.")
                st_lottie(no_default_animation, height=300, key="no_default")
            
            # Display disclaimer after prediction
            st.write("""
                ### Disclaimer!!!
                This application uses a predictive model to provide insights based on the data you input. Please note that the predictions are based on historical data and various assumptions. While the model is designed to be as accurate as possible, no prediction can be 100% accurate.

                *We recommend using the predictions as guidance and supplementing them with additional research and analysis when making decisions.*
                """)

        except ValueError as e:
            st.error(f"Error making prediction: {e}")

    if st.button("Logout"):
        st.session_state['logged_in'] = False

    # Show disclaimer outside of prediction section as well
    st.write("""
        ### Disclaimer!!!
        This application uses a predictive model to provide insights based on the data you input. Please note that the predictions are based on historical data and various assumptions. While the model is designed to be as accurate as possible, no prediction can be 100% accurate.

        *We recommend using the predictions as guidance and supplementing them with additional research and analysis when making decisions.*
        """)

    st.markdown("© SPJ Savings and Loans, 2024")
