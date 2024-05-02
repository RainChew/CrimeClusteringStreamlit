import streamlit as st
import pickle
import numpy as np

# Load the GMM model
with open('best_gmm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define the Streamlit interface
def main():
    st.title('GMM Streamlit App')
    st.write('This app demonstrates the use of a Gaussian Mixture Model (GMM) with Streamlit.')

    # User input for data
    st.header('Input Data')
    data_input = st.text_input('Enter comma-separated data points (e.g., 1.2, 3.4, 5.6):')

    if st.button('Predict'):
        # Process input data
        try:
            data = np.array([float(x.strip()) for x in data_input.split(',')])
            # Predict using the loaded model
            prediction = loaded_model.predict(data.reshape(1, -1))
            st.success(f'Cluster Prediction: {prediction[0]}')
        except:
            st.error('Error processing input. Please enter valid data.')

if __name__ == '__main__':
    main()

