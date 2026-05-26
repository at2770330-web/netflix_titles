import pickle
import pandas as pd
import streamlit as st
from pathlib import Path
import re

MODEL_PATH = Path('netflix_pipeline.pkl')

def sanitize_feature_name(value: str) -> str:
    return re.sub(r'[^0-9a-zA-Z]+', '_', value.lower()).strip('_')

def load_pipeline():
    with MODEL_PATH.open('rb') as f:
        return pickle.load(f)

def build_input_row(meta, type_ch, year, dur, genres, country, director, cast, title, desc):
    return pd.DataFrame([{
        'type': type_ch,
        'release_year': year,
        'duration_num': dur,
        'num_cast': cast,
        'genre_count': max(1, len(genres)),
        'country_top': country if country in meta['top_countries'] else 'Other',
        'director_top': director if director in meta['top_directors'] else 'Other',
        'title_len': len(title or ''),
        'description_len': len(desc or ''),
        **{f'genre_{sanitize_feature_name(g)}': int(g in genres) for g in meta['top_genres']}
    }])

def main():
    data = load_pipeline()
    st.title('Netflix Rating Prediction')
    
    col1, col2 = st.columns(2)
    with col1:
        type_ch = st.selectbox('Content Type', ['Movie', 'TV Show'])
        dur = st.slider('Duration (min)', 1, 500, 90)
        year = st.slider('Release Year', 1940, 2024, 2021)
    with col2:
        genres = st.multiselect('Genres', data['top_genres'], default=[data['top_genres'][0]])
        country = st.selectbox('Country', data['top_countries'] + ['Other'])
        director = st.selectbox('Director', data['top_directors'] + ['Other'])
    
    cast = st.number_input('Cast Members', 0, 20, 3)
    title = st.text_input('Title', 'Example')
    desc = st.text_area('Description', 'Synopsis')

    if st.button('Predict'):
        inp = build_input_row(data, type_ch, year, dur, genres, country, director, cast, title, desc)
        pred = data['pipeline'].predict(inp)[0]
        proba = data['pipeline'].predict_proba(inp)[0]
        classes = data['pipeline'].named_steps['model'].classes_
        st.success(f'Rating: **{pred}**')
        st.bar_chart(pd.Series(proba, index=classes).sort_values(ascending=False))

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
