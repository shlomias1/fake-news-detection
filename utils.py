from langdetect import detect
from deep_translator import GoogleTranslator
import re
from urllib.parse import urlparse
import pandas as pd
import hashlib
import config 

def detect_lang(text):
    return detect(text)

def translate(text, source_lang='auto', target_lang='en'):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text)

def extract_author(title):
    match = re.match(r'^([\w\s\-]+):', title)
    return match.group(1).strip() if match else ''

def extract_date_from_url(url):
    if pd.isnull(url) or not isinstance(url, str):
        return ''
    try:
        match = re.search(r'/(\d{4})/(\d{2})/(\d{2})', url)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    except Exception:
        pass
    return ''

def extract_url_from_text(text):
    if pd.isnull(text):
        return ''
    urls = re.findall(r'(https?://[^\s]+)', text)
    return urls[0] if urls else ''

def extract_date_from_url(url):
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})', url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return ''

url_parser = lambda x: urlparse(x).netloc if pd.notnull(x) else ''

hash = lambda x: hashlib.md5(x.encode()).hexdigest()

def reorder_df(df):
    for col in config.column_order:
        if col not in df.columns:
            df[col] = ''
    df = df[config.column_order]
    return df
