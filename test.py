import streamlit as st
import pandas as pd
import difflib
import joblib

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

df = load_data()

# Load saved model
vectorizer = joblib.load('tfidf_vectorizer.joblib')
cosine_sim_matrix = joblib.load('cosine_similarity_matrix.joblib')

selected_features = ['title', 'authors', 'categories', 'published_year']
for feature in selected_features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return f"{row['title']} {row['authors']} {row['categories']} {str(row['published_year'])}"  
df['combined_features'] = df.apply(combine_features, axis=1)

# Recommendation function using saved model
def get_recommendations(book_title, cosine_sim=cosine_sim_matrix, dataframe=df):
    all_titles = dataframe['title'].tolist()
    close_matches = difflib.get_close_matches(book_title, all_titles)
    if not close_matches:
        return []
    closest_match = close_matches[0]
    book_index = dataframe[dataframe.title == closest_match].index[0]
    similarity_scores = list(enumerate(cosine_sim[book_index]))
    sorted_similar_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for index, score in sorted_similar_books:
        if index != book_index:
            recommendations.append(dataframe.iloc[index]['title'])
        if len(recommendations) >= 10:
            break
    return recommendations

# Streamlit UI
st.title('Book Recommendation System (Saved Model)')
st.write('Enter a book name to get recommendations using the pre-trained model.')

book_name = st.text_input('Enter your favourite book name:')

if book_name:
    recs = get_recommendations(book_name)
    if recs:
        st.subheader(f'Recommendations for "{book_name}":')
        for i, title in enumerate(recs, 1):
            st.write(f'{i}. {title}')
    else:
        st.warning('No close match found. Please try another book.')
