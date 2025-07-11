# app.py
from recommend import df, recommend_songs
import os
import streamlit as st

base_dir = os.path.dirname(__file__)
required_files = [
    os.path.join(base_dir, 'df_cleaned.pkl'),
    os.path.join(base_dir, 'tfidf_matrix.pkl'),
    os.path.join(base_dir, 'cosine_sim.pkl')
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    import subprocess
    import logging

    logging.warning(f"Missing files: {missing_files}. Running preprocess.py...")
    try:
        subprocess.run(["python", os.path.join(base_dir, "preprocess.py")], check=True)
    except Exception as e:
        logging.error(f"Failed to run preprocess.py: {str(e)}")
        st.error("Preprocessing failed. Check logs.")



# Set custom Streamlit page config
st.set_page_config(
    page_title="Lyrics-Based Music Recommender",
    page_icon="🎶",  # Changed icon to a music note for variety
    initial_sidebar_state="expanded"  # Sidebar will be open by default
)


st.title("🎧 Discover Songs with Similar Vibes")

song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song)
        if recommendations is None:
            st.warning("Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)
