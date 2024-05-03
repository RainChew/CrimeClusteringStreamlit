import pathlib
import streamlit as st
import altair as alt
import pickle
import pandas
import numpy



def pickler(filename: str, mode: str = 'unpickle'):
    """
    Pickles the file to filename, or unpickles and returns the file

    Parameters
    ----------
    filename : str
        File to pickle to or unpickle from
    mode : str, optional
        One of 'pickle' or 'unpickle', by default 'unpickle'
    """
    if mode == 'pickle':
        pickle.dump(all_viz_data, open(filename, 'wb'))
    elif mode == 'unpickle':
        try:
            return pickle.load(open(filename, 'rb'))
        except FileNotFoundError:
            st.error(f"File {filename} not found.")
            return None

if __name__ == "__main__":
    all_viz_data = pickler(filename=pathlib.Path("df_sampled.pkl"))

    if all_viz_data is not None:
        st.header("Clustered USA Gun Violence by Date")
        st.write(f"Loaded data shape: {all_viz_data.shape}")

        number_of_clusters = st.slider("Number of clusters to display", min_value=2, max_value=10, step=1, value=5)
        st.write(f"Selected number of clusters: {number_of_clusters}")

        # Filter the dataframe to show only the selected number of clusters
        source_df = all_viz_data[all_viz_data['cluster_label'] == number_of_clusters]

        # Filter the data for years 2013 to 2018 and months 1 to 12 only
        source_df = source_df[(source_df['year'] >= 2013) & (source_df['year'] <= 2018) & (source_df['month'] >= 1) & (source_df['month'] <= 12)]

        st.write(f"Filtered data shape: {source_df.shape}")

        if not source_df.empty:
            # Generate Altair chart
            chart = alt.Chart(source_df).mark_circle(size=20).encode(
                x='month',
                y='year',
                color=alt.Color('cluster_label:N', scale=alt.Scale(scheme='category20')),
                tooltip=['cluster_label', 'month', 'year']
            ).configure_axis(
                grid=False
            ).configure_view(
                strokeWidth=0
            ).properties(
                width=600,
                height=400,
            ).interactive()

            # Display Altair chart
            st.altair_chart(chart)
        else:
            st.write("No data to display.")


    

