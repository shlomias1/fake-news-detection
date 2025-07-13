import nltk
from nltk.corpus import stopwords
import pandas as pd
import config
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import re
import polars as pl

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
    filtered_df = df.filter(pl.col(label_col) == label_value)
    word_counter = Counter()
    stop_words = set(stopwords.words('english'))
    for text in filtered_df.select(text_col).to_series():
        if text is not None:
            cleaned = clean_text(text)
            words = [w for w in cleaned.split() if w not in stop_words]
            word_counter.update(words)
    return word_counter.most_common(top_n)