import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the Streamlit interface
def main():
    st.title('GMM Streamlit App')
    st.write('This app demonstrates the use of a Gaussian Mixture Model (GMM) with Streamlit.')

    # Load your trained GMM model from the pickle file
    with open('best_gmm_model.pkl', 'rb') as f:
        best_gmm = pickle.load(f)

    # Load scatter plot data from the pickle file
    with open('scatter_plot_data.pkl', 'rb') as f:
        scatter_plot_data = pickle.load(f)

    # Extract data from the loaded dictionary
    X_pca = scatter_plot_data['X_pca']
    labels = scatter_plot_data['labels']
    cluster_labels = scatter_plot_data['cluster_labels']
    best_num_components = scatter_plot_data['best_num_components']

    # Mapping of cluster numbers to descriptive labels
    cluster_mapping = {
        1: 'Balanced Gender and Age',
        2: 'Female Dominant',
        3: 'Teen Dominant',
        4: 'Adult Dominant',
        5: 'Child Dominant',
        6: 'Adult and Teen Mix',
        7: 'Male Dominant',
        8: 'Adult Mix',
        9: 'Adult Balanced',
        10: 'Teen Mix',
        11: 'Child Mix',
        12: 'Child and Teen Mix',
        13: 'Adult and Child Mix',
        14: 'Adult and Female Mix',
        15: 'Adult and Male Mix',
        16: 'Teen and Child Mix',
        17: 'Male and Female Mix',
        18: 'Male and Child Mix',
        19: 'Female and Child Mix'
    }

    # User input for data
    st.header('Data Input')
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    st.write('Or input data manually:')
    male_count = st.number_input('Male Count', value=0)
    female_count = st.number_input('Female Count', value=0)
    teen_count = st.number_input('Teen Count', value=0)
    adult_count = st.number_input('Adult Count', value=0)
    child_count = st.number_input('Child Count', value=0)

    new_data = pd.DataFrame({
        'male_count': [male_count],
        'female_count': [female_count],
        'teen_count': [teen_count],
        'adult_count': [adult_count],
        'child_count': [child_count]
    })

    if st.button('Predict'):
        # Predict using the loaded model
        prediction = best_gmm.predict(new_data)
        cluster_label = cluster_mapping.get(prediction[0], 'Unknown')
        st.success(f'Cluster Prediction: {cluster_label}')

        # Generate scatter plot
        scatter_plot(X_pca, labels, cluster_labels, best_num_components)

def scatter_plot(X_pca, labels, cluster_labels, best_num_components):
    # Plot PCA-transformed data with cluster labels
    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster_label in range(best_num_components):
        ax.scatter(X_pca[labels == cluster_label, 0], X_pca[labels == cluster_label, 1], label=cluster_labels[cluster_label])
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.set_title('PCA Scatter Plot with GMM Clusters')
    ax.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()
