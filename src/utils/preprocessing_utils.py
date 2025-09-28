import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
from urllib.parse import urlparse
import pandas as pd
import hashlib
import src.config as config

def extract_author(title):
    match = re.match(r'^([\w\s\-]+):', title)
    return match.group(1).strip() if match else 'unknown'

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

def extract_title_from_url(url):
    if not isinstance(url, str) or not url:
        return ''
    path = urlparse(url).path  
    last_part = path.strip('/').split('/')[-1]
    title = re.sub(r'[-_]+', ' ', last_part)
    return title.strip()

url_parser = lambda x: urlparse(x).netloc if pd.notnull(x) else ''

def generate_hash(index: int) -> str:
    return hashlib.md5(str(index).encode()).hexdigest()

def reorder_df(df):
    for col in config.column_order:
        if col not in df.columns:
            df[col] = ''
    df = df[config.column_order]
    return df

def extract_channel_name(aria_label):
    if pd.isnull(aria_label):
        return ''
    
    match = re.search(r'by (.+?) \d', aria_label)
    return match.group(1).strip() if match else ''

def classify_channel_risk(channel_name):
    if not channel_name:
        return 0.5
    trusted_channels = [
        'CNN', 'ABC News', 'BBC', 'CBS', 'TIME', 'The Telegraph', 'Inside Edition', 'CBC', 'HISTORY'
    ]
    satire_or_fake_channels = [
        'The Hummus News', 'Sportspickle', 'Scrappleface', 'Reductress', 'Reel Truth History Documentaries'
    ]
    lower_name = channel_name.lower()
    if any(trusted.lower() in lower_name for trusted in trusted_channels):
        return 1.0
    elif any(fake.lower() in lower_name for fake in satire_or_fake_channels):
        return 0.0
    else:
        return 0.5
    
def _label_verdict(t):
    if t in ["fake", "satire", "unreliable", "clickbait", "rumor", "conspiracy", "hate", "bias", "junksci"]:
        return 0
    elif t in ["reliable"]:
        return 1
    else:
        return -1

def add_default_fields(df, defaults: dict):
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df

def rename_columns(df, rename_map: dict):
    return df.rename(columns=rename_map)
