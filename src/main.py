# app.py

import os
import subprocess
import streamlit as st

# Set base directory path (to src/)
base_dir = os.path.dirname(__file__)

# List of expected .pkl files
required_files = [
    os.path.join(base_dir, 'df_cleaned.pkl'),
    os.path.join(base_dir, 'tfidf_matrix.pkl'),
    os.path.join(base_dir, 'cosine_sim.pkl')
]

# Check for missing files and run preprocessing if needed
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.warning("🔄 Required files not found. Running preprocessing...")
    try:
        result = subprocess.run(
            ["python", os.path.join(base_dir, "preprocess.py")],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("✅ Preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        st.error("❌ Preprocessing failed.")
        st.error(e.stderr or e.stdout or str(e))
        raise

from recommend import df, recommend_songs

# Set custom Streamlit page config
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",  # You can also use a path to a .ico or .png file
    layout="centered"
)


st.title("🎶 Instant Music Recommender")

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
