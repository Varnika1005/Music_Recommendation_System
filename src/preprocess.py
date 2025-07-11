import os
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

nltk.download('punkt')
nltk.download('stopwords')

# Compute path to dataset
base_dir = os.path.dirname(__file__)  # path to /src
csv_path = os.path.join(base_dir, "../spotify_millsongdata.csv")

# Load and sample dataset
try:
    df = pd.read_csv(csv_path).sample(10000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Preprocessing
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# TF-IDF
logging.info("🔠 Vectorizing...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

# Cosine similarity
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save outputs
joblib.dump(df, os.path.join(base_dir, 'df_cleaned.pkl'))
joblib.dump(tfidf_matrix, os.path.join(base_dir, 'tfidf_matrix.pkl'))
joblib.dump(cosine_sim, os.path.join(base_dir, 'cosine_sim.pkl'))

logging.info("✅ Preprocessing complete and files saved.")
