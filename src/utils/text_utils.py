import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import nltk
from nltk.corpus import stopwords
import pandas as pd
import src.config as config
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import re
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

def remove_stop_words(text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    if pd.isnull(text):
        return ''
    words = text.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered)

def remove_hebrew_stop_words(text):
    if pd.isnull(text):
        return ''
    words = text.split()
    filtered = [w for w in words if w not in config.hebrew_stop_words]
    return ' '.join(filtered)

def detect_lang(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None

def parallel_translate(df, text_col, lang_col, output_col, target_lang='en', max_workers=10):
    def translate_single(row):
        text = row[text_col]
        source_lang = row[lang_col]
        if pd.isnull(text) or source_lang == target_lang:
            return text
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"[Translation Error] {str(text)[:30]}... - {e}")
            return text
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(translate_single, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Translating {output_col}"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Thread Error] Row {idx}: {e}")
                results[idx] = df.loc[idx, text_col]
    df[output_col] = results
    return df

def translate(df, text_col, lang_col, output_col, target_lang='en'):
    def translate_single(row):
        text = row[text_col]
        source_lang = row[lang_col]
        if pd.isnull(text) or source_lang == target_lang:
            return text
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"[Translation Error] {str(text)[:30]}... - {e}")
            return text
    tqdm.pandas(desc=f"Translating {output_col}")
    df[output_col] = df.progress_apply(translate_single, axis=1)
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_top_words(df, label_value, label_col='label', text_col='text', top_n=25):
    nltk.download('stopwords')
    filtered_df = df.filter(pl.col(label_col) == label_value)
    word_counter = Counter()
    stop_words = set(stopwords.words('english'))
    for text in filtered_df.select(text_col).to_series():
        if text is not None:
            cleaned = clean_text(text)
            words = [w for w in cleaned.split() if w not in stop_words]
            word_counter.update(words)
    return word_counter.most_common(top_n)

def remove_boilerplate(text: str) -> str:
    for phrase in config.BOILERPLATE_PHRASES:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text

def extract_bigram(df: pl.DataFrame, text_col: str, min_df=1, max_df=1.0, ngram_range=(2, 2)):
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    df = df.with_columns([
        pl.col("text")
        .map_elements(remove_boilerplate)
        .alias("cleaned_text")
    ])
    texts_raw = df[text_col].to_list()
    texts = [t if isinstance(t, str) and t.strip() else "" for t in texts_raw]
    vectorizer = CountVectorizer(analyzer='word',
                                 ngram_range=ngram_range,
                                 min_df=min_df,
                                 max_df=max_df,
                                 stop_words='english')
    X = vectorizer.fit_transform(texts)
    sum_words = X.sum(axis=0)
    bigrams_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
    for bigram, freq in bigrams_freq[:20]:
        print(f"{bigram}: {freq}")
    return bigrams_freq