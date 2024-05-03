import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define the Streamlit interface
def main():
    scaler = StandardScaler()
    

    st.title('KMeans Clustering Streamlit App')
    st.write('This app demonstrates KMeans clustering with Streamlit.')

    # User input for latitude, longitude, and total number of victims
    st.header('Location Input')
    latitude = st.number_input('Latitude', value=0.0, step=0.0001)
    longitude = st.number_input('Longitude', value=0.0, step=0.0001)
    total_victims = st.number_input('Total Number of Victims', value=0, step=1)

    # Load the KMeans model and DataFrame from the pickle file
    with open('kmeans_model_and_data.pkl', 'rb') as f:
        kmeans, df = pickle.load(f)

    if st.button('Predict'):
        # Scale the input data
        # X_scaled = scaler.fit_transform([[latitude, longitude, total_victims]])
        scaled_input = scaler.fit_transform([[latitude, longitude, total_victims]])

        # Predict cluster label for the scaled input
        cluster_label = predict_cluster(scaled_input, kmeans)

        # Display cluster label
        st.success(f'Predicted Cluster Label: {cluster_label}')

def predict_cluster(scaled_input, kmeans):
    # Predict cluster label for the input location
    cluster_label = kmeans.predict(scaled_input)
    return cluster_label[0]

if __name__ == '__main__':
    main()
