import streamlit as st
import hashlib
import numpy as np
import pickle

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

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
        # Home ownership dropdown
        home_ownership = st.selectbox("Home Ownership Status", ("Own", "Mortgage", "Rent"))

    # Create another row with 3 columns for Employment Duration, Loan Purpose, and Loan Amount
    col4, col5, col6 = st.columns(3)

    with col4:
        # Employment Duration
        employment_duration = st.number_input("Employment Duration (Years)", min_value=0, value=0)

    with col5:
        # Purpose of the loan dropdown
        loan_purpose = st.selectbox("Purpose of the Loan", ("Education", "Home Improvement"))

    with col6:
        # Loan applied
        loan_applied = st.number_input("Loan Applied (Dollars)", min_value=0.0, step=500.0, value=0.0)
    
    col7, col8, col9 = st.columns(3)
    with col7:
        rate = st.number_input("Rate (%)", min_value=0.0, step=0.1, value=0.0)
    with col8:
        default = st.selectbox("Default",("Yes","No"))
    with col9:
        credit_history = st.number_input("Credit History", min_value=0.0, step=1.0, value=0.0)

    # Button for prediction
    if st.button("Predict"):
        # Preprocess the input data (like during model training)
        # 1. Map home_ownership to numeric values
        home_ownership_mapping = {"Own": 1, "Mortgage": 2, "Rent": 3}
        home_ownership_value = home_ownership_mapping[home_ownership]

        # 2. Map loan_purpose to numeric values
        loan_purpose_mapping = {"Education": 1, "Home Improvement": 2}
        loan_purpose_value = loan_purpose_mapping[loan_purpose]

        # 3. Map default to numeric
        default_value = 1 if default == "Yes" else 0

        # 4. Construct the input array (needs to be in the same format as training)
        input_data = np.array([[age, annual_income, employment_duration, loan_applied, rate, 
                                credit_history, home_ownership_value, loan_purpose_value, default_value]])

        # 5. Scale the numerical features
        numerical_features_indices = [1, 3, 4, 5]  # Adjust indices based on the input order
        input_data[:, numerical_features_indices] = scaler.transform(input_data[:, numerical_features_indices])

        # 6. Make the prediction using the model
        prediction = model.predict(input_data)

        # 7. Output the result
        if prediction[0] == 1:
            st.success(f"The model predicts that {client_name} will default.")
        else:
            st.success(f"The model predicts that {client_name} will not default.")

    # Back button to log out and return to login page
    if st.button("Back to Login"):
        st.session_state['logged_in'] = False
        st.success("You have logged out. Please log in again.")
