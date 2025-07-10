import nltk
from nltk.corpus import stopwords
import pandas as pd
import config
from langdetect import detect
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    return detect(text)

def translate(df, text_col, lang_col, output_col, target_lang='en', max_workers=10):
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