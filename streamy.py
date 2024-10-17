import streamlit as st
import hashlib
import numpy as np
import pickle
import requests
import io

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
model_file_id = "1kOA_SQUh9FQydycChQrWIBgMty6xG5EW"  # Model file ID
#scaler_file_id = "1kNF4X9rEADxhvnnvpG1pheMjwNGZcErq"  # Scaler file ID

# Load the trained model and scaler from Google Drive
model_content = download_file_from_google_drive(model_file_id)
if model_content:
    try:
        model = pickle.load(io.BytesIO(model_content))
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()  # Stop the execution if model fails to load
else:
    st.stop()  # Stop the execution if the model cannot be loaded

#scaler_content = download_file_from_google_drive(scaler_file_id)
#if scaler_content:
    #try:
       # scaler = pickle.load(io.BytesIO(scaler_content))
        #st.success("Scaler loaded successfully.")
   # except Exception as e:
       # st.error(f"Error loading scaler: {e}")
       # st.stop()  # Stop the execution if scaler fails to load
#else:
    #st.stop()  # Stop the execution if the scaler cannot be loaded

# Function to hash the password (though not used in this case)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Title for the application
st.title("Welcome to SPJ Savings and Loans")

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
    st.title("SPJ Savings and Loans Predictive Model")

    # Instructions
    st.write("Please fill in the details below")

    # Client Information
    st.header("Client Information")

    # Name of Client
    client_name = st.text_input("Name of Client")

    # Create 3 columns for Age, Annual Income, and Home Ownership Status
    col1, col2, col3 = st.columns(3)

    with col1:
        # Age input
        age = st.number_input("Age (18 and above)", min_value=18, value=18)

    with col2:
        # Annual income input
        annual_income = st.number_input("Annual Income (Dollars)", min_value=0.0, step=1000.0, value=0.0)

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
        loan_applied = st.number_input("Loan Applied (Dollars)", min_value=0.0, step=500.0, value=0.0)
    
    col7, col8, col9 = st.columns(3)
    with col7:
        rate = st.number_input("Rate (%)", min_value=0.0, step=0.1, value=0.0)
    with col8:
        default = st.selectbox("Default", ("", "Yes", "No"))
    with col9:
        credit_history = st.number_input("Credit History", min_value=0.0, step=1.0, value=0.0)

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
        if prediction[0] == 1:
            st.success(f"The model predicts that {client_name} **WILL DEFAULT**.\nThis is the second line")
            st.write("""
### Disclaimer: Predictive Model
This application uses a **predictive model** to provide insights based on the data you input. Please note that the predictions are based on historical data and various assumptions. While the model is designed to be as accurate as possible, **no prediction can be 100% accurate**.

We recommend using the predictions as guidance and supplementing them with additional research and analysis when making decisions.
""")
        else:
            st.success(f"The model predicts that {client_name} **WILL NOT DEFAULT**")
            st.write("""
### Disclaimer: Predictive Model
This application uses a **predictive model** to provide insights based on the data you input. Please note that the predictions are based on historical data and various assumptions. While the model is designed to be as accurate as possible, **no prediction can be 100% accurate**.

We recommend using the predictions as guidance and supplementing them with additional research and analysis when making decisions.
""")

    # Back button to log out and return to login page
    if st.button("Back to Login"):
        st.session_state['logged_in'] = False
        st.success("You have logged out. Please log in again.")
