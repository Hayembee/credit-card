import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('LogisticRegression.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess input data
def preprocess_data(data):
    # Assuming 'data' is a pandas DataFrame
    # You may need to adapt this based on your actual data preprocessing steps
    scaler = StandardScaler()

    # Extract relevant columns (only 'Time' and 'Amount')
    selected_columns = ['Time', 'Amount']

    # If data is empty, return an array of zeros
    if data.empty:
        return np.zeros((1, len(selected_columns)))

    # Select relevant columns and scale
    data_scaled = scaler.fit_transform(data[selected_columns])
    return data_scaled

# Streamlit app
def main():
    st.title('Credit Fraud Detection App')

    st.sidebar.header('User Input Features')

    # Define the feature names (only 'Time' and 'Amount')
    feature_names = ['Time', 'Amount']

    # Initialize session state
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []

    # Collect input features from the user
    user_input = {}
    for feature in feature_names:
        if feature == 'Time':
            # Assuming 'Time' is in minutes
            user_input[feature] = st.sidebar.number_input(f'{feature} (minutes)', value=0, help=f'Explanation of {feature}...')
        elif feature == 'Amount':
            # Assuming 'Amount' is in dollars
            user_input[feature] = st.sidebar.number_input(f'{feature} (dollars)', value=0.0, help=f'Explanation of {feature}...')

    # Add the user input to transactions when the "Add Transaction" button is clicked
    if st.sidebar.button('Add Transaction'):
        st.session_state.transactions.append(user_input)

    # Display transactions
    st.subheader('Transactions')
    transactions_df = pd.DataFrame(st.session_state.transactions)
    st.write(transactions_df)

    # Make predictions and check for potentially fraudulent transactions when the user clicks the "Detect Fraud" button
    if st.button('Detect Fraud'):
        # Preprocess the input data
        input_data = preprocess_data(transactions_df)

        # Make predictions
        #prediction = model.predict(input_data)
        #prediction_proba = model.predict_proba(input_data)

        # Display prediction result
        st.subheader('Prediction')
        # Check for potentially fraudulent transactions
        if len(transactions_df) <= 5:
            time_interval_threshold = 30  # 30 minutes
            if (
                transactions_df['Time'].diff().fillna(0).max() > time_interval_threshold
            ):
                st.warning('No fraudulent activity detected.')

        elif len(transactions_df) >= 3:
            time_interval_threshold = 2  # 2 minutes
            
            if (
                transactions_df['Time'].diff().fillna(0).max() <= time_interval_threshold
            ):
                st.warning('Potential fraudulent activity detected, block credit card')

        # Display prediction probability
        # st.subheader('Prediction Probability')
        # st.write(f'Probability of Fraud: {prediction_proba[0][1]:.2%}')

if __name__ == '__main__':
    main()
